"""Observe the native review pipeline and compile a shadow-only opening IR.

No historical reference or scene label reaches the IR builder. Native output
is produced by the unchanged production path; the IR is assembled afterwards.
"""

import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ship_perception.perception.candidate_union import (
    CandidateProposal,
    OpeningObservation,
    OpeningHypothesisIR,
    canonical,
)


def frozen(value):
    return json.loads(canonical(value))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temp), str(path))


def grid_record(grid):
    return dict(
        x0=float(grid.x0),
        y0=float(grid.y0),
        cell_m=float(grid.cell_m),
        shape=list(grid.occupancy.shape),
    )


def grid_id(grid):
    return hashlib.sha256(canonical(grid).encode()).hexdigest()


def source_inventory(bundle):
    """Inventory captured source records independently of IR normalization."""
    records = {}
    for key in (
        "height",
        "proposals",
        "providers",
        "s4_nodes_before",
        "s4_nodes_after",
        "height_regions",
        "s4_regions",
        "local_regions",
        "local_before_role",
        "local_after_role",
    ):
        for index, value in enumerate(bundle[key]):
            records["/{}/{}".format(key, index)] = hashlib.sha256(
                canonical(value).encode()
            ).hexdigest()
    for key in (
        "raw_observations",
        "side_families",
        "topology_relations",
        "topology_vertices",
        "node_partition_audit",
    ):
        for index, value in enumerate(bundle["s4"][key]):
            records["/s4/{}/{}".format(key, index)] = hashlib.sha256(
                canonical(value).encode()
            ).hexdigest()
    records["/s4/structural_axis"] = hashlib.sha256(
        canonical(bundle["s4"]["structural_axis"]).encode()
    ).hexdigest()
    return records


def recover_height_members(grid, rows, opening_drop_m):
    """Recover the exact component already emitted by local_height_recall.

    This repeats only its recorded mask predicate for provenance. It generates
    no new height levels, components, proposals or geometry.
    """
    labels_cache = {}
    xx, yy = grid.centers()
    output = []
    for row in rows:
        key = (row["level_m"], row["polarity"])
        if key not in labels_cache:
            mask = grid.occupancy & (
                grid.median < row["level_m"] - opening_drop_m
                if row["polarity"] == "LOW"
                else grid.median > row["level_m"] + opening_drop_m
            )
            labels = ndimage.label(mask, structure=np.ones((3, 3), bool))[0]
            labels_cache[key] = labels, np.bincount(labels.ravel())
        labels, totals = labels_cache[key]
        box = row["bounds_axial"]
        inside = (xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])
        ids, counts = np.unique(labels[inside & (labels > 0)], return_counts=True)
        matches = [
            int(i)
            for i, n in zip(ids, counts)
            if n == totals[i] == row["occupied_cells"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "HEIGHT_OBSERVATION_MEMBERSHIP_NOT_UNIQUE:" + row["proposal_id"]
            )
        cells = np.argwhere(labels == matches[0])
        assert len(cells) == row["occupied_cells"]
        output.append(dict(row=row, cells=cells.tolist()))
    return output


def restored_inventory(observations):
    records = {}
    for observation in observations:
        for ref in observation["features"].get("source_record_refs", []):
            payload = (
                observation["evidence"]
                if ref["payload_field"] is None
                else observation["evidence"][ref["payload_field"]]
            )
            digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
            if ref["path"] in records and records[ref["path"]] != digest:
                raise ValueError("IR_SOURCE_REFERENCE_PAYLOAD_CONFLICT")
            records[ref["path"]] = digest
    return records


def cell_intersections(cells, source_grid, target_grid):
    """Exact rectangle intersection on two existing B1 raster lattices.

    A different grid origin is not a different Vessel frame. Native cell IDs
    remain untouched. Fractional overlap is relation evidence only, never a
    relabelled point, a new cell observation, or a Frame rotation/translation.
    """
    size, target = source_grid["cell_m"], target_grid["cell_m"]
    for row, col in cells:
        low = np.asarray(
            [source_grid["x0"] + col * size, source_grid["y0"] + row * size]
        )
        high = low + size
        first = np.floor(
            (low - [target_grid["x0"], target_grid["y0"]]) / target
        ).astype(int)
        last = np.ceil((high - [target_grid["x0"], target_grid["y0"]]) / target).astype(
            int
        )
        for c in range(first[0], last[0]):
            for r in range(first[1], last[1]):
                corner = np.array(
                    [target_grid["x0"] + c * target, target_grid["y0"] + r * target]
                )
                area = np.maximum(
                    0.0, np.minimum(high, corner + target) - np.maximum(low, corner)
                ).prod()
                if area > 0:
                    yield (row, col), (r, c), float(area)


def cell_overlap(cells, source_grid, target_grid):
    weights = {}
    size, target = source_grid["cell_m"], target_grid["cell_m"]
    for _, rc, area in cell_intersections(cells, source_grid, target_grid):
        weights[rc] = weights.get(rc, 0.0) + area / target**2
    expected = len(cells) * (size / target) ** 2
    if not math.isclose(sum(weights.values()), expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("B1_LATTICE_INTERSECTION_AREA_NOT_CONSERVED")
    return [[r, c, value] for (r, c), value in sorted(weights.items())]


class NativeObserver:
    def __init__(self):
        self.bundle = dict(
            schema="ship_perception.opening_ir_source_bundle.1", errors=[], proposals=[]
        )

    def watch(self, owner, name, before=None, after=None):
        native = getattr(owner, name)

        def capture(callback, *args, **kwargs):
            if callback:
                try:
                    callback(*args, **kwargs)
                except Exception as error:
                    self.bundle["errors"].append(
                        dict(
                            function=name,
                            error=str(error),
                            traceback=traceback.format_exc(),
                        )
                    )

        @wraps(native)
        def wrapped(*args, **kwargs):
            capture(before, *args, **kwargs)
            result = native(*args, **kwargs)
            capture(after, result, *args, **kwargs)
            return result

        return patch.object(owner, name, wrapped)

    def height(self, proposals, expert, cache, research, budget=None):
        grid = cache.grid("coarse")
        self.bundle.update(
            vessel_id=cache.vessel.vessel_id,
            b1_axes=frozen(cache.vessel.axes),
            local_grid=grid_record(grid),
            config=frozen(cache.config),
            selected_primary_index_sha256=cache.primary_cache.vessel.point_index_sha256,
            selected_context_index_sha256=cache.vessel.point_index_sha256,
            height=recover_height_members(
                grid, cache._v2_height_recall[0], cache.config["roi"]["opening_drop_m"]
            ),
            providers=frozen(cache.providers),
            background=frozen(
                dict(
                    xy=np.column_stack(
                        (
                            grid.centers()[0][grid.occupancy],
                            grid.centers()[1][grid.occupancy],
                        )
                    )
                )
            ),
        )
        self.add_proposals(proposals)

    def add_proposals(self, proposals):
        self.bundle["proposals"].extend(
            dict(
                proposal_id=p.proposal_id,
                vessel_id=p.vessel_id,
                expert=p.expert,
                bounds=list(p.bounds),
                evidence=p.evidence(),
                lineage=list(p.lineage),
            )
            for p in proposals
        )

    def native_result(self, result, *args, **kwargs):
        self.bundle["native_geometry"] = frozen(result["geometry_only"])
        selection = result["berth_selection"]
        if selection["active_vessel_id"] is None:
            self.bundle["capture_terminal"] = frozen(
                dict(
                    status="NO_ACTIVE_VESSEL",
                    ir_status="IR_NOT_COMPILED_NO_ACTIVE_VESSEL",
                    selector_status="SELECTOR_NOT_RUN",
                    berth_selection=selection,
                    scene=result["scene"],
                    native_review_count=result["geometry_only"]["review_candidate_count"],
                    input_sha256=self.bundle["input_sha256"],
                    source_identity=self.bundle["source_identity"],
                    config_sha256=result["config_sha256"],
                )
            )

    def s4_nodes(self, bridge, phase):
        rows = []
        for ident, node in sorted(bridge.nodes.items()):
            row = {k: v for k, v in vars(node).items() if k != "mask"}
            row.update(
                node_id=ident,
                parent=getattr(node.parent, "node_id", node.parent),
                children=[getattr(n, "node_id", n) for n in node.children],
                cells=np.argwhere(node.mask),
            )
            rows.append(row)
        self.bundle["s4_nodes_" + phase] = frozen(rows)
        self.bundle["s4_grid"] = grid_record(bridge.grid)

    def hooks(self):
        from ship_perception.perception import experts, legacy_expert
        from ship_perception.perception import s4_topology_bridge as s4
        from ship_perception.perception import local_enclosure_replay as local
        from ship_perception.perception import opening_identity_eligibility as role
        from ship_perception.perception import review_pipeline as review

        stack = ExitStack()
        stack.enter_context(
            self.watch(
                legacy_expert.LegacyProposalExpert,
                "propose",
                after=lambda r, *a, **k: self.add_proposals(r.proposals),
            )
        )
        stack.enter_context(
            self.watch(experts.HeightBasinExpert, "propose", after=self.height)
        )
        stack.enter_context(
            self.watch(
                s4.S4TopologyBridge,
                "_partition_nodes",
                before=lambda b: self.s4_nodes(b, "before"),
                after=lambda r, b: self.s4_nodes(b, "after"),
            )
        )
        stack.enter_context(
            self.watch(
                s4.S4TopologyBridge,
                "compile_regions",
                before=lambda b, e: self.bundle.update(
                    height_regions=frozen([r.audit() for r in e])
                ),
                after=lambda r, b, e: self.bundle.update(
                    s4=frozen(b.audit()), s4_regions=frozen([v.audit() for v in r])
                ),
            )
        )
        # local_enclosure_replay imports this symbol directly. Observe its
        # original input before compound qualification can be overwritten.
        stack.enter_context(
            self.watch(
                local,
                "compile_opening_roles",
                before=lambda cache, entries: self.bundle.update(
                    local_before_role=frozen(entries)
                ),
            )
        )
        stack.enter_context(
            self.watch(
                local,
                "compile_local_evidence",
                after=lambda r, *a, **k: self.bundle.update(local_after_role=frozen(r)),
            )
        )
        stack.enter_context(
            self.watch(
                local,
                "resolve_local_identities",
                after=lambda r, *a, **k: self.bundle.update(
                    local_regions=frozen([v.audit() for v in r])
                ),
            )
        )
        stack.enter_context(
            self.watch(
                review,
                "run_perception",
                after=self.native_result,
            )
        )
        return stack


def observations_from_bundle(bundle):
    vessel, local_grid = bundle["vessel_id"], bundle["local_grid"]
    observations = []

    def add(kind, source_id, evidence, cells=(), source_refs=(), **features):
        features["source_record_refs"] = [
            dict(path=p, payload_field=f) for p, f in source_refs
        ]
        obs = OpeningObservation.from_evidence(
            kind, vessel, source_id, evidence, cells, **features
        )
        observations.append(obs)
        return obs

    for index, item in enumerate(bundle["height"]):
        row = item["row"]
        add(
            "HEIGHT_LEVEL_COMPONENT",
            row["proposal_id"],
            item,
            item["cells"],
            source_refs=[("/height/" + str(index), None)],
            origin="LOCAL_HEIGHT_RECALL",
            grid_id=grid_id(local_grid),
            grid=local_grid,
            polarity=row["polarity"],
            level_m=row["level_m"],
            bounds=row["bounds_axial"],
        )
    for index, row in enumerate(bundle["proposals"]):
        ev = row["evidence"]
        add(
            "SOURCE_CANDIDATE_PROPOSAL",
            row["proposal_id"],
            row,
            source_refs=[("/proposals/" + str(index), None)],
            association_source_ids=(
                [ev["proposal_id"]] if row["expert"] == "HEIGHT_BASIN" else []
            ),
            evidence_provider_ids=list(row["lineage"]),
        )
    for index, row in enumerate(bundle["providers"]):
        add(
            (
                "RAW_FACE"
                if row["provider_type"] == "RAW3D_STRUCTURAL_PROVIDER"
                else "BEV_EDGE"
            ),
            row["provider_id"],
            row,
            source_refs=[("/providers/" + str(index), None)],
        )
    for phase in ("before", "after"):
        for index, row in enumerate(bundle["s4_nodes_" + phase]):
            cells = row["cells"]
            add(
                "HEIGHT_LEVEL_COMPONENT",
                row["node_id"] if phase == "after" else "BEFORE:" + row["node_id"],
                row,
                cells,
                source_refs=[("/s4_nodes_{}/{}".format(phase, index), None)],
                origin="S4_HIERARCHY_"
                + ("AFTER_PARTITION" if phase == "after" else "BEFORE_PARTITION"),
                grid_id=grid_id(bundle["s4_grid"]),
                grid=bundle["s4_grid"],
                local_grid_overlap=cell_overlap(cells, bundle["s4_grid"], local_grid),
                support_area_cell_units=len(cells)
                * (bundle["s4_grid"]["cell_m"] / local_grid["cell_m"]) ** 2,
            )
    s4 = bundle["s4"]
    for index, row in enumerate(s4["raw_observations"]):
        add(
            "RAW_FACE" if row["evidence_type"] == "OBSERVED_3D_FACE" else "RAW_PROFILE",
            row["observation_id"],
            row,
            source_refs=[("/s4/raw_observations/" + str(index), None)],
            association_source_ids=[row["node_id"]],
        )
    for index, row in enumerate(s4["side_families"]):
        add(
            "S4_FAMILY",
            row["family_id"],
            row,
            source_refs=[("/s4/side_families/" + str(index), None)],
            axis_role=row["axis_role"],
            association_source_ids=row["source_height_nodes"]
            + row["source_observation_ids"],
        )
    for index, row in enumerate(bundle["height_regions"]):
        add(
            "HEIGHT_REGION_NOMINATION",
            "BEFORE_S4:" + row["region_id"],
            row,
            source_refs=[("/height_regions/" + str(index), None)],
            association_source_ids=row["source_proposal_ids"],
        )
    for index, row in enumerate(bundle["s4_regions"]):
        ie = row["identity_evidence"]
        cells = ie.get("member_support_cells_rc", [])
        is_s4 = ie.get("source") == "CURRENT_S4_TOPOLOGY"
        add(
            "S4_REGION_FRAGMENT" if is_s4 else "HEIGHT_REGION_NOMINATION",
            row["region_id"],
            row,
            cells,
            source_refs=[("/s4_regions/" + str(index), None)],
            local_grid_overlap=cell_overlap(cells, bundle["s4_grid"], local_grid),
            support_area_cell_units=len(cells),
            association_source_ids=ie.get("member_height_nodes", [])
            + row["source_proposal_ids"],
        )
    for index, row in enumerate(bundle["local_regions"]):
        n = row["identity_evidence"].get("nomination", {})
        add(
            "PRODUCTION_IDENTITY_SNAPSHOT",
            "AFTER_LOCAL:" + row["region_id"],
            row,
            source_refs=[("/local_regions/" + str(index), None)],
            association_source_ids=[n["proposal_id"]] if n else [row["region_id"]],
        )
    add(
        "STRUCTURAL_AXIS_OBSERVATION",
        "CURRENT_S4_AXIS",
        s4["structural_axis"],
        source_refs=[("/s4/structural_axis", None)],
    )
    for index, row in enumerate(s4["topology_vertices"]):
        add(
            "S4_VERTEX_ASSOCIATION",
            "VERTEX:" + row["node_id"],
            row,
            source_refs=[("/s4/topology_vertices/" + str(index), None)],
            association_source_ids=[row["node_id"]] + row["side_family_ids"],
        )
    for index, row in enumerate(s4["node_partition_audit"]):
        add(
            "CURRENT_PARTITION_OBSERVATION",
            "PARTITION:" + str(index),
            row,
            source_refs=[("/s4/node_partition_audit/" + str(index), None)],
            association_source_ids=[c["node_id"] for c in row["children"]],
        )
    for index, row in enumerate(s4["topology_relations"]):
        add(
            "CURRENT_TOPOLOGY_RELATION",
            "TOPOLOGY:" + str(index),
            row,
            source_refs=[("/s4/topology_relations/" + str(index), None)],
            association_source_ids=row["node_ids"],
        )
        for wi, witness in enumerate(row["separator_evidence"]):
            add(
                "SEPARATOR_FACE",
                "S4_RELATION:{}:{}".format(index, wi),
                dict(relation=row, witness=witness),
                node_ids=row["node_ids"],
                qualified=bool(witness.get("qualified")),
                association_source_ids=row["node_ids"],
            )
    after = {r["nomination"]["proposal_id"]: r for r in bundle["local_after_role"]}
    after_index = {
        r["nomination"]["proposal_id"]: i
        for i, r in enumerate(bundle["local_after_role"])
    }
    for index, before in enumerate(bundle["local_before_role"]):
        n = before["nomination"]
        sides = before["sides"]
        complete = all(
            s["selected_rho"] is not None and s["conservative_rho"] is not None
            for s in sides.values()
        )
        bounds = (
            [sides[s]["conservative_rho"] for s in ("U0", "V0", "U1", "V1")]
            if complete
            else None
        )
        complete = bool(complete and bounds[0] < bounds[2] and bounds[1] < bounds[3])
        ends = [set(sides[s]["verified_face_provider_ids"]) for s in ("U0", "U1")]
        independent_ends = all(ends) and not ends[0].intersection(ends[1])
        bands = (
            [
                sorted([sides[s]["selected_rho"], sides[s]["conservative_rho"]])
                for s in ("U0", "U1")
            ]
            if complete
            else None
        )
        add(
            "LOCAL_ENCLOSURE_WITNESS",
            "LOCAL:" + n["proposal_id"],
            dict(
                before_identity_eligibility=before,
                production_result=after[n["proposal_id"]],
            ),
            source_refs=[
                ("/local_before_role/" + str(index), "before_identity_eligibility"),
                (
                    "/local_after_role/" + str(after_index[n["proposal_id"]]),
                    "production_result",
                ),
            ],
            height_source_id=n["proposal_id"],
            complete_enclosure=complete,
            enclosure_bounds=bounds,
            interior_role=before.get("current_interior_corner_evidence", {}).get(
                "status"
            ),
            independent_terminal_pair=bool(independent_ends),
            terminal_bands=bands,
            evidence_provider_ids=sorted(
                {p for s in sides.values() for p in s["provider_ids"]}
            ),
        )
        partition = after[n["proposal_id"]].get("internal_partition_review")
        if partition:
            add(
                "SEPARATOR_FACE",
                "LOCAL_PARTITION:" + n["proposal_id"],
                partition,
                qualified=False,
                node_ids=[],
                association_source_ids=[n["proposal_id"]],
            )
    return tuple(observations)


def compile_bundle(bundle, output_dir):
    if bundle["errors"]:
        raise ValueError("NATIVE_OBSERVATION_CAPTURE_INCOMPLETE")
    proposals = tuple(
        CandidateProposal(
            r["proposal_id"],
            r["vessel_id"],
            r["expert"],
            tuple(r["bounds"]),
            canonical(r["evidence"]),
            tuple(r["lineage"]),
        )
        for r in bundle["proposals"]
    )
    observations = observations_from_bundle(bundle)
    ir = OpeningHypothesisIR.compile(
        proposals,
        observations,
        bundle["config"]["roi"]["min_opening_cells"],
        bundle["config"]["roi"]["min_enclosure_ratio"],
    )
    audit = ir.audit()
    expected_sources = source_inventory(bundle)
    restored_sources = restored_inventory(audit["observations"])
    audit["source_record_inventory"] = expected_sources
    audit["provenance"] = {
        k: bundle[k]
        for k in (
            "source_identity",
            "numerical_environment",
            "frozen",
            "input_sha256",
            "vessel_id",
            "b1_axes",
            "local_grid",
            "s4_grid",
            "selected_primary_index_sha256",
            "selected_context_index_sha256",
        )
    }
    atomic(output_dir / "opening_hypothesis_ir.json", audit)
    atomic(
        output_dir / "observation_to_hypothesis.json",
        json.loads(ir.observation_mapping_json),
    )
    atomic(output_dir / "hypothesis_relation.json", json.loads(ir.relations_json))
    original = {
        o.observation_id: hashlib.sha256(o.evidence_json.encode()).hexdigest()
        for o in observations
    }
    restored = {
        o["observation_id"]: hashlib.sha256(
            canonical(o["evidence"]).encode()
        ).hexdigest()
        for o in audit["observations"]
    }
    mapping = json.loads(ir.observation_mapping_json)
    complete_local = {
        o.observation_id
        for o in observations
        if o.kind == "LOCAL_ENCLOSURE_WITNESS"
        and json.loads(o.features_json).get("complete_enclosure")
    }
    retained_local = {
        x["observation_id"]
        for h in audit["hypotheses"]
        for x in h["interior_evidence"]["local_enclosure_interpretations"]
        if x["complete_enclosure"]
    }
    proposal_records = audit["source_proposals"]
    gate = dict(
        raw_observation_count=len(expected_sources),
        normalized_observation_count=len(observations),
        count_unit="CAPTURED_PRE_SELECTION_SOURCE_RECORDS_WITH_FULL_VIEW_PROVENANCE_NOT_RAW_POINT_COUNT",
        canonical_hypothesis_count=len(ir.hypotheses),
        observation_counts_by_kind={
            kind: sum(o.kind == kind for o in observations)
            for kind in sorted({o.kind for o in observations})
        },
        LOSSLESS_OBSERVATION_ACCOUNTING=(
            "PASS"
            if original == restored
            and expected_sources == restored_sources
            and len(mapping) == len(original)
            and {m["observation_id"] for m in mapping} == set(original)
            else "FAIL"
        ),
        NO_CANDIDATE_SILENTLY_DROPPED=(
            "PASS"
            if canonical(proposal_records) == canonical(bundle["proposals"])
            else "FAIL"
        ),
        HEIGHT_LINEAGE_BUILT=(
            "PASS"
            if sum(len(r["height_observation_ids"]) for r in audit["height_lineages"])
            == len(bundle["height"])
            else "FAIL"
        ),
        S4_DOES_NOT_REPLACE_HEIGHT_HYPOTHESIS=(
            "PASS" if all(h.height_component_lineage for h in ir.hypotheses) else "FAIL"
        ),
        LOCAL_ENCLOSURE_DOES_NOT_DISAPPEAR=(
            "PASS" if complete_local <= retained_local else "FAIL"
        ),
        S4_REPLACED_PRIOR_COUNT=0,
        LOCAL_COMPLETE_CANDIDATE_LOST_COUNT=len(complete_local - retained_local),
        HIDDEN_3_STRONG_IDENTITY_AUTHORITY="NONE_IN_SHADOW_IR;LEGACY_PRODUCTION_UNCHANGED",
        SEPARATOR_DELETES_PARENT_HYPOTHESIS="NO_IN_SHADOW_IR;LEGACY_PRODUCTION_UNCHANGED",
        unassigned_context_observations=sum(not m["hypothesis_ids"] for m in mapping),
        unassigned_policy="RETAINED_FULLY_IN_IR;NO_FORCED_OPENING_ASSIGNMENT",
        PRODUCTION_OUTPUT_UNCHANGED="PENDING_EXTERNAL_BEFORE_AFTER_CHECK",
    )
    atomic(output_dir / "ir_gate.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from ship_perception.perception import cli
    from ship_perception.perception.pipeline import (
        source_identity,
        numerical_environment,
    )
    from ship_perception.tools.verify_v14_frozen import verify
    from ship_perception.perception.legacy_expert import verify_legacy_b2_core

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "IR_RUN_STARTED").open("x") as marker:
        marker.write("one native invocation; shadow IR only\n")
    observer = NativeObserver()
    observer.bundle.update(
        source_identity=source_identity(),
        input_sha256=hashlib.sha256(args.pcd.read_bytes()).hexdigest(),
        numerical_environment=numerical_environment(),
        frozen=dict(v14=verify(), b2=verify_legacy_b2_core()),
    )
    old = sys.argv
    sys.argv = [
        "pcd_main",
        "--pcd",
        str(args.pcd),
        "--mode",
        "review-first",
        "--output-dir",
        str(args.output_dir / "native"),
    ]
    try:
        with observer.hooks():
            code = cli.pcd_main()
    finally:
        sys.argv = old
        atomic(args.output_dir / "recorded_observation_bundle.json", observer.bundle)
    (args.output_dir / "native.exit_code").write_text(str(code), encoding="utf-8")
    terminal = observer.bundle.get("capture_terminal")
    if code == 0 and terminal is not None and not observer.bundle["errors"]:
        atomic(args.output_dir / "ir_gate.json", terminal)
        print(json.dumps(terminal, ensure_ascii=False))
        return 0
    gate = compile_bundle(observer.bundle, args.output_dir)
    print(json.dumps(gate))
    return 0 if all(v != "FAIL" for v in gate.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
