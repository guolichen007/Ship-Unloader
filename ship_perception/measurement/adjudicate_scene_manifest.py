"""Compare independent raw-PCD reviews; never decide geometry on a human's behalf.

All review coordinates and outputs belong under ignored Ship-Unloader-Data/ or
Ship-Unloader-Work/. A signed decision file and final scene manifest are required
before this tool emits a scoring audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from ship_perception.measurement.blind_review_pack import NORMAL, sha
from ship_perception.measurement.oracle_local_geometry import inside_xy
from ship_perception.tools.validate_v15r_scene_manifest import DATASET, SCHEMA, read_json, validate

EXPECTED = dict(zip(NORMAL, (2, 1, 1, 1, 1, 3)))
REVIEW_SCHEMA = Path(__file__).with_name("r0a_review.schema.json")


def polygon(value, name):
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"MISSING_POLYGON:{name}")
    array = np.asarray(value, dtype=float)
    if array.shape != (len(value), 2) or not np.isfinite(array).all():
        raise ValueError(f"INVALID_POLYGON:{name}")
    area = abs(np.sum(array[:, 0] * np.roll(array[:, 1], -1) -
                      array[:, 1] * np.roll(array[:, 0], -1))) / 2
    if area < 1e-4:
        raise ValueError(f"DEGENERATE_POLYGON:{name}")
    return array


def polyline(value, name):
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"MISSING_EDGE_SUPPORT:{name}")
    array = np.asarray(value, dtype=float)
    if array.shape != (len(value), 2) or not np.isfinite(array).all():
        raise ValueError(f"INVALID_EDGE_SUPPORT:{name}")
    if np.linalg.norm(np.diff(array, axis=0), axis=1).sum() < 1e-4:
        raise ValueError(f"ZERO_LENGTH_EDGE:{name}")
    return array


def sample_line(points, step=0.05):
    parts = []
    for a, b in zip(points, points[1:]):
        length = np.linalg.norm(b-a)
        if length:
            parts.append(a + np.linspace(0, 1, max(2, int(np.ceil(length/step))+1))[:, None]*(b-a))
    return np.vstack(parts)


def point_segment_distance(points, chain):
    best = np.full(len(points), np.inf)
    for a, b in zip(chain, chain[1:]):
        d = b-a
        if d @ d:
            t = np.clip(((points-a) @ d)/(d @ d), 0, 1)
            best = np.minimum(best, np.linalg.norm(points-(a+t[:, None]*d), axis=1))
    return best


def edge_difference(a, b):
    a, b = polyline(a, "review_A"), polyline(b, "review_B")
    distances = np.r_[point_segment_distance(sample_line(a), b),
                      point_segment_distance(sample_line(b), a)]
    return {"p50_m": float(np.percentile(distances, 50)),
            "p95_m": float(np.percentile(distances, 95))}


def polygon_difference(a, b, cell=0.25):
    a, b = polygon(a, "review_A"), polygon(b, "review_B")
    low = np.minimum(a.min(axis=0), b.min(axis=0))
    high = np.maximum(a.max(axis=0), b.max(axis=0))
    sizes = np.maximum(1, np.ceil((high-low)/cell).astype(int))
    if int(np.prod(sizes)) > 1_000_000:
        raise ValueError("POLYGON_COMPARISON_TOO_LARGE")
    xs, ys = np.meshgrid(low[0]+(np.arange(sizes[0])+.5)*cell,
                         low[1]+(np.arange(sizes[1])+.5)*cell)
    xy = np.column_stack((xs.ravel(), ys.ravel()))
    aa, bb = inside_xy(xy, a), inside_xy(xy, b)
    union = np.count_nonzero(aa | bb)
    return {"polygon_iou_approx": float(np.count_nonzero(aa & bb)/union) if union else 0.0,
            "centroid_distance_m": float(np.linalg.norm(a.mean(axis=0)-b.mean(axis=0)))}


def validate_review(review, *, role, scan, index_sha):
    sid = scan["scan_id"]
    issues = list(Draft202012Validator(read_json(REVIEW_SCHEMA)).iter_errors(review))
    if issues:
        raise ValueError(f"REVIEW_SCHEMA_INVALID:{role}:{sid}:{issues[0].message}")
    if review.get("schema") != "ship_perception.r0a.human_review.1" or review.get("review_status") != "COMPLETE":
        raise ValueError(f"REVIEW_NOT_COMPLETE:{role}:{sid}")
    for field, expected in (("scan_id", sid), ("source_pcd_sha256", scan["pcd_sha256"]),
                            ("blind_pack_index_sha256", index_sha), ("reviewer_role", role)):
        if review.get(field) != expected:
            raise ValueError(f"REVIEW_BINDING_MISMATCH:{role}:{sid}:{field}")
    if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip():
        raise ValueError(f"MISSING_REVIEWER_ID:{role}:{sid}")
    for field in ("repo_access", "review_a_prior_access", "algorithm_output_access", "raw_render_only"):
        if not isinstance(review.get(field), bool):
            raise ValueError(f"MISSING_REVIEW_PROVENANCE:{role}:{sid}:{field}")
    if not review["raw_render_only"]:
        raise ValueError(f"ALGORITHM_OUTPUT_LEAKAGE:{role}:{sid}")
    if role == "B" and (review["repo_access"] or review["review_a_prior_access"] or
                        review["algorithm_output_access"]):
        raise ValueError(f"REVIEW_B_NOT_INDEPENDENT:{sid}")
    polygon(review.get("vessel_roi_raw_xy"), f"{sid}:vessel")
    count = review.get("visible_in_fov_hatch_count")
    hatches = review.get("hatches")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0 or not isinstance(hatches, list) or count != len(hatches):
        raise ValueError(f"REVIEW_HATCH_COUNT_MISMATCH:{role}:{sid}")
    ids = set()
    for h in hatches:
        hid = h.get("local_id")
        if not isinstance(hid, str) or not hid or hid in ids:
            raise ValueError(f"INVALID_HATCH_ID:{role}:{sid}")
        ids.add(hid)
        polygon(h.get("rough_polygon_raw_xy"), f"{sid}:{hid}")
        if h.get("observation_status") not in {"VISIBLE", "PARTIAL", "UNRESOLVED"}:
            raise ValueError(f"INVALID_HATCH_OBSERVATION:{role}:{sid}:{hid}")
        edges = h.get("edges")
        if not isinstance(edges, list):
            raise ValueError(f"MISSING_EDGES_LIST:{role}:{sid}:{hid}")
        eids = set()
        for edge in edges:
            eid = edge.get("edge_local_id")
            if not isinstance(eid, str) or not eid or eid in eids:
                raise ValueError(f"INVALID_EDGE_ID:{role}:{sid}:{hid}")
            eids.add(eid)
            if edge.get("visibility") not in {"VISIBLE", "PARTIAL", "UNCERTAIN", "NOT_IN_FOV"}:
                raise ValueError(f"INVALID_EDGE_VISIBILITY:{role}:{sid}:{hid}:{eid}")
            if edge.get("opening_side_status") not in {"OBSERVABLE", "AMBIGUOUS_TRANSITION", "NOT_OBSERVABLE"}:
                raise ValueError(f"INVALID_OPENING_SIDE_STATUS:{role}:{sid}:{hid}:{eid}")
            support = edge.get("observed_support")
            if support is not None:
                polyline(support, f"{sid}:{hid}:{eid}")
            if edge.get("opening_side_status") == "OBSERVABLE" and edge["visibility"] == "VISIBLE" and support is None:
                raise ValueError(f"VISIBLE_EDGE_WITHOUT_SUPPORT:{role}:{sid}:{hid}:{eid}")
            if not isinstance(edge.get("reason"), str):
                raise ValueError(f"MISSING_EDGE_REASON:{role}:{sid}:{hid}:{eid}")


def review_pair(a, b, scan, index_sha):
    validate_review(a, role="A", scan=scan, index_sha=index_sha)
    validate_review(b, role="B", scan=scan, index_sha=index_sha)
    if a["reviewer_id"] == b["reviewer_id"]:
        raise ValueError(f"SAME_REVIEWER:{scan['scan_id']}")
    rows = []
    if len(a["hatches"]) != len(b["hatches"]):
        status = "COUNT_DISAGREEMENT"
    else:
        status = "PAIRING_REQUIRES_HUMAN_CONFIRMATION"
    pairs = sorted((polygon_difference(x["rough_polygon_raw_xy"], y["rough_polygon_raw_xy"])["polygon_iou_approx"],
                    x["local_id"], y["local_id"])
                   for x in a["hatches"] for y in b["hatches"])
    used_a, used_b = set(), set()
    for overlap, aid, bid in reversed(pairs):
        if aid in used_a or bid in used_b:
            continue
        used_a.add(aid); used_b.add(bid)
        ah = next(h for h in a["hatches"] if h["local_id"] == aid)
        bh = next(h for h in b["hatches"] if h["local_id"] == bid)
        a_edges = [e for e in ah["edges"] if e["observed_support"] is not None]
        b_edges = [e for e in bh["edges"] if e["observed_support"] is not None]
        epairs = sorted((edge_difference(x["observed_support"], y["observed_support"])["p95_m"],
                         x["edge_local_id"], y["edge_local_id"])
                        for x in a_edges for y in b_edges)
        used_ae, used_be, edge_rows = set(), set(), []
        for _, aeid, beid in epairs:
            if aeid in used_ae or beid in used_be:
                continue
            used_ae.add(aeid); used_be.add(beid)
            ae = next(e for e in a_edges if e["edge_local_id"] == aeid)
            be = next(e for e in b_edges if e["edge_local_id"] == beid)
            diff = edge_difference(ae["observed_support"], be["observed_support"])
            edge_rows.append({"review_A_edge_id": aeid, "review_B_edge_id": beid,
                              "review_A_visibility": ae["visibility"], "review_B_visibility": be["visibility"],
                              "review_A_side": ae["opening_side_status"],
                              "review_B_side": be["opening_side_status"],
                              "inter_reviewer_p50_m": diff["p50_m"],
                              "inter_reviewer_p95_m": diff["p95_m"],
                              "gt_15cm_compatible_provisional": diff["p95_m"] <= .05 and
                              all(e["visibility"] == "VISIBLE" and e["opening_side_status"] == "OBSERVABLE"
                                  for e in (ae, be)),
                              "human_pairing_confirmed": False})
        rows.append({"review_A_hatch_id": aid, "review_B_hatch_id": bid,
                     **polygon_difference(ah["rough_polygon_raw_xy"], bh["rough_polygon_raw_xy"]),
                     "suggested_edge_pairs_not_decisions": edge_rows,
                     "unpaired_review_A_edges": sorted(set(e["edge_local_id"] for e in ah["edges"])-used_ae),
                     "unpaired_review_B_edges": sorted(set(e["edge_local_id"] for e in bh["edges"])-used_be),
                     "human_pairing_confirmed": False})
    return {"scan_id": scan["scan_id"], "status": status,
            "review_A_count": len(a["hatches"]), "review_B_count": len(b["hatches"]),
            "unpaired_A": sorted(set(h["local_id"] for h in a["hatches"])-used_a),
            "unpaired_B": sorted(set(h["local_id"] for h in b["hatches"])-used_b),
            "suggested_pairs_not_decisions": rows}


def load_reviews(review_root, blind_index):
    dataset = {s["scan_id"]: s for s in read_json(DATASET)["scans"]}
    index = read_json(blind_index)
    if index.get("annotation_inputs_read") is not False or index.get("algorithm_outputs_read") is not False:
        raise ValueError("BLIND_PACK_NOT_RAW_ONLY")
    records = {r["scan_id"]: r for r in index["scans"]}
    if set(records) != set(NORMAL):
        raise ValueError("BLIND_PACK_SCAN_SET_MISMATCH")
    for sid in NORMAL:
        if records[sid]["pcd_sha256"] != dataset[sid]["pcd_sha256"]:
            raise ValueError(f"BLIND_PACK_PCD_HASH_MISMATCH:{sid}")
        for render in records[sid]["renders"]:
            filename = render["filename"]
            if Path(filename).name != filename or sha(blind_index.parent/sid/filename) != render["sha256"]:
                raise ValueError(f"BLIND_RENDER_HASH_MISMATCH:{sid}:{filename}")
    index_sha = sha(blind_index)
    reviews = {}
    for sid in NORMAL:
        reviews[sid] = {}
        for role in ("A", "B"):
            path = review_root/f"review_{role}"/f"{sid}.json"
            reviews[sid][role] = (read_json(path), sha(path))
        review_pair(reviews[sid]["A"][0], reviews[sid]["B"][0], dataset[sid], index_sha)
    return dataset, reviews, index_sha


def compare(review_root: Path, blind_index: Path, output: Path):
    dataset, reviews, index_sha = load_reviews(review_root, blind_index)
    report = {"schema": "ship_perception.r0a.disagreement.1", "blind_pack_index_sha256": index_sha,
              "note": "Suggested polygon pairing is not adjudication; physical edges require explicit human links.",
              "scans": []}
    for sid in NORMAL:
        a, b = reviews[sid]["A"][0], reviews[sid]["B"][0]
        row = review_pair(a, b, dataset[sid], index_sha)
        row["review_provenance"] = {role: {"reviewer_id": reviews[sid][role][0]["reviewer_id"],
                                           "review_file_sha256": reviews[sid][role][1],
                                           "repo_access": reviews[sid][role][0]["repo_access"],
                                           "review_a_prior_access": reviews[sid][role][0]["review_a_prior_access"]}
                                    for role in ("A", "B")}
        report["scans"].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    decision = {"schema": "ship_perception.r0a.human_decision.1", "adjudicator_id": None,
                "note": "Human must confirm each pairing and manually author final manifest; no auto-adjudication.",
                "scans": [{"scan_id": row["scan_id"], "review_A_sha256": row["review_provenance"]["A"]["review_file_sha256"],
                           "review_B_sha256": row["review_provenance"]["B"]["review_file_sha256"],
                           "roi_confirmed": False,
                           "hatch_links": [{"a": pair["review_A_hatch_id"], "b": pair["review_B_hatch_id"],
                                            "final": None, "human_confirmed": False}
                                           for pair in row["suggested_pairs_not_decisions"]],
                           "edge_links": [{"final_hatch": None, "final_edge": None,
                                           "a_edge": edge["review_A_edge_id"],
                                           "b_edge": edge["review_B_edge_id"],
                                           "opening_side_human_confirmed": False}
                                          for pair in row["suggested_pairs_not_decisions"]
                                          for edge in pair["suggested_edge_pairs_not_decisions"]]}
                          for row in report["scans"]]}
    (output.parent/"human_decision_DRAFT.json").write_text(
        json.dumps(decision, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    draft = {"schema_version": "ship_perception.v15r.scene_manifest.2", "annotation_status": "DRAFT",
             "scans": [{"scan_id": sid, "source_pcd_sha256": dataset[sid]["pcd_sha256"],
                        "split_role": "DEVELOPMENT", "scene_review_status": "NEEDS_REVIEW",
                        "operational_target_vessel_id": "target", "reviewers": [reviews[sid]["A"][0]["reviewer_id"],
                                                                                 reviews[sid]["B"][0]["reviewer_id"]],
                        "independent_reviews": [{"reviewer_id": reviews[sid][role][0]["reviewer_id"],
                                                 "review_file_sha256": reviews[sid][role][1]} for role in ("A", "B")],
                        "vessels": [{"vessel_local_id": "target", "vessel_role": "PRIMARY_DEVELOPMENT",
                                     "roi_raw_xy": None, "fov_coverage": "UNKNOWN",
                                     "visible_in_fov_hatch_count": None, "whole_vessel_hatch_count": None,
                                     "count_certainty": "UNREVIEWED", "subdivision_evidence": [], "hatches": [],
                                     "reason": "Human adjudicator must fill ROI, hatches, observed edges."}],
                        "reason": "DRAFT; no automatic semantic decisions."} for sid in NORMAL]}
    (output.parent/"scene_manifest_DRAFT.json").write_text(
        json.dumps(draft, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return report


def verify_final(manifest_path: Path, decision_path: Path, review_root: Path, blind_index: Path, audit_path: Path):
    dataset, reviews, index_sha = load_reviews(review_root, blind_index)
    manifest = read_json(manifest_path)
    errors, ready = validate(manifest, read_json(DATASET), read_json(SCHEMA), True)
    if errors or set(ready) != set(NORMAL) or set(s["scan_id"] for s in manifest["scans"]) != set(NORMAL):
        raise ValueError("FINAL_MANIFEST_INVALID:"+";".join(errors))
    decision = read_json(decision_path)
    if decision.get("schema") != "ship_perception.r0a.human_decision.1" or not decision.get("adjudicator_id"):
        raise ValueError("MISSING_HUMAN_DECISION_SIGNOFF")
    decisions = {s["scan_id"]: s for s in decision.get("scans", [])}
    if set(decisions) != set(NORMAL):
        raise ValueError("DECISION_SCAN_SET_MISMATCH")
    scans = {s["scan_id"]: s for s in manifest["scans"]}
    edge_eligibility = {}
    for sid in NORMAL:
        scan, dec = scans[sid], decisions[sid]
        a, ahash = reviews[sid]["A"]
        b, bhash = reviews[sid]["B"]
        if len(a["hatches"]) != EXPECTED[sid] or len(b["hatches"]) != EXPECTED[sid]:
            raise ValueError(f"REVIEWER_COUNT_DISAGREEMENT_OR_EXPECTED_MISMATCH:{sid}")
        if scan.get("reviewers") != [a["reviewer_id"], b["reviewer_id"]] or scan.get("independent_reviews") != [
                {"reviewer_id": a["reviewer_id"], "review_file_sha256": ahash},
                {"reviewer_id": b["reviewer_id"], "review_file_sha256": bhash}]:
            raise ValueError(f"FINAL_REVIEW_PROVENANCE_MISMATCH:{sid}")
        if dec.get("review_A_sha256") != ahash or dec.get("review_B_sha256") != bhash or dec.get("roi_confirmed") is not True:
            raise ValueError(f"DECISION_REVIEW_BINDING_MISMATCH:{sid}")
        vessel = [v for v in scan["vessels"] if v["vessel_role"] == "PRIMARY_DEVELOPMENT"]
        if len(vessel) != 1 or vessel[0].get("roi_raw_xy") is None or len(vessel[0]["hatches"]) != EXPECTED[sid]:
            raise ValueError(f"FINAL_ROI_OR_COUNT_MISMATCH:{sid}")
        final_hatches = {h["local_id"]: h for h in vessel[0]["hatches"]}
        links = dec.get("hatch_links", [])
        if len(links) != EXPECTED[sid] or len({r["final"] for r in links}) != len(links) or \
                len({r["a"] for r in links}) != len(links) or len({r["b"] for r in links}) != len(links):
            raise ValueError(f"DECISION_HATCH_PAIRING_INCOMPLETE:{sid}")
        for link in links:
            if link.get("human_confirmed") is not True or link["final"] not in final_hatches or \
                    link["a"] not in {h["local_id"] for h in a["hatches"]} or \
                    link["b"] not in {h["local_id"] for h in b["hatches"]}:
                raise ValueError(f"DECISION_HATCH_PAIRING_INVALID:{sid}")
            if final_hatches[link["final"]]["rough_polygon_raw_xy"] is None:
                raise ValueError(f"FINAL_HATCH_POLYGON_MISSING:{sid}:{link['final']}")
        edge_links = dec.get("edge_links", [])
        if (len({(r.get("final_hatch"), r.get("final_edge")) for r in edge_links}) != len(edge_links) or
                len({(r.get("final_hatch"), r.get("a_edge")) for r in edge_links}) != len(edge_links) or
                len({(r.get("final_hatch"), r.get("b_edge")) for r in edge_links}) != len(edge_links)):
            raise ValueError(f"DECISION_EDGE_LINK_NOT_ONE_TO_ONE:{sid}")
        for edge_link in edge_links:
            hid, eid = edge_link["final_hatch"], edge_link["final_edge"]
            if hid not in final_hatches:
                raise ValueError(f"EDGE_LINK_UNKNOWN_HATCH:{sid}:{hid}")
            final_edges = {e["edge_local_id"]: e for e in final_hatches[hid]["edges"]}
            if eid not in final_edges:
                raise ValueError(f"EDGE_LINK_UNKNOWN_FINAL_EDGE:{sid}:{hid}:{eid}")
            hlink = next(r for r in links if r["final"] == hid)
            ae = next((e for h in a["hatches"] if h["local_id"] == hlink["a"] for e in h["edges"]
                       if e["edge_local_id"] == edge_link.get("a_edge")), None)
            be = next((e for h in b["hatches"] if h["local_id"] == hlink["b"] for e in h["edges"]
                       if e["edge_local_id"] == edge_link.get("b_edge")), None)
            if ae is None or be is None or ae["observed_support"] is None or be["observed_support"] is None:
                raise ValueError(f"EDGE_REVIEW_SUPPORT_MISSING:{sid}:{hid}:{eid}")
            difference = edge_difference(ae["observed_support"], be["observed_support"])
            final = final_edges[eid]
            eligible = bool(edge_link.get("opening_side_human_confirmed") is True and difference["p95_m"] <= 0.05 and
                            all(e["visibility"] == "VISIBLE" and e["opening_side_status"] == "OBSERVABLE"
                                for e in (ae, be, final)) and final["observed_support"] is not None)
            edge_eligibility[f"{sid}/{hid}/{eid}"] = {"status": "SCORABLE" if eligible else "EDGE_GT_UNRESOLVED",
                                                       "inter_reviewer_p50_m": difference["p50_m"],
                                                       "inter_reviewer_p95_m": difference["p95_m"],
                                                       "reviewer_A_support": ae["observed_support"],
                                                       "reviewer_B_support": be["observed_support"]}
    all_final_edges = sum(len(h["edges"]) for s in scans.values() for v in s["vessels"]
                          if v["vessel_role"] == "PRIMARY_DEVELOPMENT" for h in v["hatches"])
    scorable = [v for v in edge_eligibility.values() if v["status"] == "SCORABLE"]
    gt_p95 = float(np.percentile([v["inter_reviewer_p95_m"] for v in edge_eligibility.values()], 95)) if edge_eligibility else None
    audit = {"schema": "ship_perception.r0a.scoring_audit.1", "manifest_sha256": sha(manifest_path),
             "decision_sha256": sha(decision_path), "blind_pack_index_sha256": index_sha,
             "decision_file": str(decision_path.resolve()), "blind_index_file": str(blind_index.resolve()),
             "review_files": {sid: {role: {"path": str((review_root/f"review_{role}"/f"{sid}.json").resolve()),
                                            "sha256": reviews[sid][role][1]} for role in ("A", "B")}
                              for sid in NORMAL},
             "adjudicator_id": decision["adjudicator_id"], "proposal_gt_ready": True,
             "normal_scene_count": len(NORMAL), "adjudicated_hatch_count": sum(EXPECTED.values()),
             "physical_gt_edge_total": all_final_edges, "physical_gt_edge_scorable": len(scorable),
             "gt_inter_review_p95_m": gt_p95,
             "gt_edge_p95_compatible_with_15cm_gate": gt_p95 is not None and gt_p95 <= .05 and
                                                        len(scorable) == len(edge_eligibility),
             "physical_edges": edge_eligibility}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return audit


def require_audit(manifest_path: Path):
    path = manifest_path.with_name("adjudication_audit.json")
    audit = read_json(path)
    if audit.get("schema") != "ship_perception.r0a.scoring_audit.1" or audit.get("manifest_sha256") != sha(manifest_path) or \
            audit.get("proposal_gt_ready") is not True or audit.get("normal_scene_count") != len(NORMAL) or \
            audit.get("adjudicated_hatch_count") != sum(EXPECTED.values()):
        raise ValueError("ADJUDICATION_AUDIT_MISSING_OR_STALE")
    if sha(audit["decision_file"]) != audit["decision_sha256"] or \
            sha(audit["blind_index_file"]) != audit["blind_pack_index_sha256"] or \
            set(audit["review_files"]) != set(NORMAL):
        raise ValueError("ADJUDICATION_SOURCE_STALE")
    for sid in NORMAL:
        for role in ("A", "B"):
            row = audit["review_files"][sid][role]
            if sha(row["path"]) != row["sha256"]:
                raise ValueError(f"REVIEW_FILE_SHA_MISMATCH:{sid}:{role}")
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-root", type=Path, required=True)
    ap.add_argument("--blind-index", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("Ship-Unloader-Work/review/r0_a/disagreement.json"))
    ap.add_argument("--final-manifest", type=Path)
    ap.add_argument("--decision-file", type=Path)
    args = ap.parse_args()
    compare(args.review_root, args.blind_index, args.output)
    if args.final_manifest or args.decision_file:
        if not args.final_manifest or not args.decision_file:
            raise ValueError("FINAL_MANIFEST_AND_HUMAN_DECISION_REQUIRED_TOGETHER")
        verify_final(args.final_manifest, args.decision_file, args.review_root, args.blind_index,
                     args.final_manifest.with_name("adjudication_audit.json"))
    print(args.output)


if __name__ == "__main__":
    main()
