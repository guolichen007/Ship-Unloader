"""Normalize review captures and match convex footprints in original XY."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_bytes()) if path.is_file() else {}


def signed_area(polygon):
    return sum(np.cross(a, b) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))) / 2


def overlap(first, second):
    """Convex polygon clipping, independent of start vertex and winding."""
    a, b = (np.asarray(p, dtype=float) for p in (first, second))
    # Work relative to a shared origin for a stable area calculation.
    origin = a[0].copy()
    a, b = a - origin, b - origin
    area_a, area_b = abs(signed_area(a)), abs(signed_area(b))
    if not area_a or not area_b:
        return None
    if signed_area(b) < 0:
        b = b[::-1]
    result = list(a)
    for left, right in zip(b, np.roll(b, -1, axis=0)):
        previous_points, result = result, []
        if not previous_points:
            break
        start = previous_points[-1]
        ds = np.cross(right - left, start - left)
        for end in previous_points:
            de = np.cross(right - left, end - left)
            if (ds >= 0) != (de >= 0):
                result.append(start + (end - start) * (ds / (ds - de)))
            if de >= 0:
                result.append(end)
            start, ds = end, de
    common = min(area_a, area_b, abs(signed_area(np.asarray(result)))) if result else 0
    return dict(iou=float(common / (area_a + area_b - common)),
                first_contained=float(common / area_a), second_contained=float(common / area_b),
                center_distance_m=float(np.linalg.norm(a.mean(0) - b.mean(0))))


def normalize(capture):
    native_files = sorted(capture.rglob("03_results.json"))
    native = read(native_files[-1]) if native_files else {}
    ir = read(capture / "opening_hypothesis_ir.json")
    shadow = read(capture / "selection.json")
    geometry = native.get("geometry_only", {})
    regions = {r["region_id"]: r for r in geometry.get("opening_regions", [])}
    selected = {r["hypothesis_id"]: r for r in shadow.get("records", [])}
    vessels = {r.get("vessel_hypothesis_id"): r for r in native.get("scene", {}).get("vessel_hypotheses", [])}
    rows = []
    for state, records in (("RECTANGLE_REVIEW", geometry.get("hatches", [])),
                           ("UNRESOLVED", geometry.get("unresolved_rectangle_candidates", []))):
        for record in records:
            h = record.get("pre_solve_candidate", record)
            key = h.get("opening_identity")
            region = regions.get(key, {})
            evidence = region.get("identity_evidence", {})
            rows.append(dict(candidate_id=key, vessel_id=region.get("vessel_id"), state=state,
                             source_kind=evidence.get("source"), source_ids=h.get("source_proposal_ids", []),
                             geometry=dict(polygon_xy=h.get("boundary_xy"), sides=h.get("sides"),
                                           nullable_z=[z.get("z") for z in h.get("corner_heights", [])] or record.get("corner_z_m")),
                             physical=dict(member_grid=evidence.get("member_support_grid"),
                                           member_grid_cell_count=len(evidence.get("member_support_cells_rc", [])),
                                           terminal_family_ids=evidence.get("opposite_terminal_family_ids"),
                                           owned_raw_index_sha=None),
                             first_failed_gate=record.get("reason"), steel_certified=False))
    for h in ir.get("hypotheses", []):
        key, vessel = h["hypothesis_id"], h.get("vessel_id")
        selection = selected.get(key, {})
        bounds, axes = h.get("conservative_bounds"), vessels.get(vessel, {}).get("local_axes")
        polygon = None
        if bounds is not None and axes is not None:
            u0, v0, u1, v1 = bounds
            polygon = (np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ axes).tolist()
        rows.append(dict(candidate_id=key, vessel_id=vessel,
                         state="PHYSICAL_SELECTED" if selection.get("selected") else "PROPOSED",
                         source_kind=h.get("status"), source_ids=h.get("source_proposal_ids", []),
                         source_region_ids=h.get("source_region_ids", []),
                         geometry=dict(polygon_xy=polygon, local_bounds_uv=bounds),
                         physical=dict(terminal_family_ids=h.get("terminal_family_ids"),
                                       separator_ids=selection.get("separator_witness_ids"),
                                       owned_raw_index_sha=None),
                         first_failed_gate=selection.get("selection_reason"), steel_certified=False))
    return dict(input_sha256=native.get("input_sha256"), producer_sha=native.get("producer_git_sha"),
                runtime_source_sha256=native.get("runtime_source_sha256"), config_sha256=native.get("config_sha256"),
                source_frame=native.get("input", {}).get("frame_id"), schemas=[native.get("schema"), ir.get("schema")],
                active_vessel=native.get("berth_selection", {}).get("active_vessel_id"),
                records=rows, structure_scope_ids=shadow.get("selected_structure_scope_ids", []))


def relations(formal, candidates):
    links = []
    for a in formal:
        for b in candidates:
            if a["candidate_id"] == b["candidate_id"] or a["vessel_id"] != b["vessel_id"]:
                continue
            p, q = a["geometry"]["polygon_xy"], b["geometry"]["polygon_xy"]
            metrics = overlap(p, q) if p is not None and q is not None else None
            direct = a["candidate_id"] in b.get("source_region_ids", [])
            if direct or (metrics and metrics["iou"] > 0):
                links.append(dict(first=a["candidate_id"], second=b["candidate_id"], direct_lineage=direct,
                                  metrics=metrics, physical_identity="UNRESOLVED_SOURCE"))
    return links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = normalize(args.capture)
    rows = result["records"]
    result["relations"] = relations([r for r in rows if r["state"] in ("RECTANGLE_REVIEW", "UNRESOLVED")], rows)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

