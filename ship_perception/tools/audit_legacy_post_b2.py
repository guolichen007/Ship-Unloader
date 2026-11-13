"""Compare read-only historical post-B2 evidence with a V2 forensic funnel.

This evaluation tool consumes archived diagnostics only. Neither the detector
nor its configuration imports it. Old candidate geometry is not a steel Golden.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path


FUNCTIONS = (
    ("perception/pipeline.py", "_legacy_proposals"),
    ("r1_static/pcd_showcase.py", "prepare_geometry"),
    ("r1_static/corner_topology_review.py", "review_points"),
    ("r1_static/corner_topology_review.py", "_separator_split"),
    ("r1_static/r2h_b5_corner_forensics.py", "_observations"),
    ("r1_static/r2h_b5_corner_forensics.py", "_side_modes"),
    ("r1_static/boundary_topology.py", "_observations"),
    ("perception/final_boundary.py", "FinalBoundaryRefiner"),
)


def source_functions(root):
    result = []
    for relative, name in FUNCTIONS:
        source = (root / "ship_perception" / relative).read_text(encoding="utf-8")
        node = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
        text = ast.get_source_segment(source, node)
        result.append(
            dict(
                file=relative,
                function=name,
                line=node.lineno,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
    return result


def semantic_summary(row):
    sides = row.get("deck_ring", {}).get("sides", {})
    return dict(
        candidate_id=row["proposal_id"],
        b2_proposal_id=row.get("prior_evidence", {}).get("hatch_hypothesis_id"),
        separator_topology=(
            "TWO_MEASURED_FACES"
            if isinstance(row.get("measured_separator_faces"), list)
            else "UNSPLIT"
        ),
        side_evidence_availability={
            name: dict(
                raw3d=bool(s.get("raw3d_support")), bev=bool(s.get("bev_support"))
            )
            for name, s in sides.items()
        },
        support_side_count=sum(
            s.get("raw3d_support", False) and s.get("bev_support", False)
            for s in sides.values()
        ),
        profile_observation_count=len(row.get("profile_observations", [])),
        profile_evidence_types=sorted(
            {p["evidence_type"] for p in row.get("profile_observations", [])}
        ),
        local_physical_candidate_available=True,
        steel_role_certified=False,
    )


def compare(old_dir, before_report, old_source, current_source):
    records = []
    for scene in before_report["scenes"]:
        path = old_dir / scene["scene"] / "post_b2.json"
        if not path.exists():
            continue
        old = json.loads(path.read_bytes())
        before = scene["funnel"]
        traces = {
            t["legacy_proposal_id"]: t
            for t in before["proposal_trace"]
            if t["legacy_proposal_id"]
        }
        candidates = []
        for row in old["post_b2_candidates"]:
            summary = semantic_summary(row)
            b2_id = summary["b2_proposal_id"]
            trace = traces.get("LEGACY:" + b2_id) if b2_id else None
            candidates.append(
                dict(
                    old=summary,
                    current_b2_trace=trace,
                    identity_relation=(
                        "B5_FORENSIC_B2_ASSIGNMENT"
                        if b2_id
                        else "NO_ONE_TO_ONE_B2_ASSIGNMENT"
                    ),
                    bounds_post_b2=row["bounds_axial"],
                    post_b2_bounds_equal_b2=bool(
                        trace and trace["original_bounds"] == row["bounds_axial"]
                    ),
                    current_post_b2_semantics_available=False,
                    blocker_status="CONFIRMED" if trace else "OPEN_LINEAGE_UNASSIGNED",
                )
            )
        records.append(
            dict(
                scene=scene["scene"],
                input_sha=old["b2"]["input_sha256"],
                old_counts=old["counts"],
                old_legacy_physical_candidate_count=len(candidates),
                current_final_count=before["final"]["hatch_count"],
                candidates=candidates,
            )
        )
    return dict(
        schema="ship_perception.legacy_post_b2_semantics_diff.1",
        usage="HISTORICAL_PATH_REGRESSION_NOT_STEEL_GOLDEN",
        old_source_functions=source_functions(old_source),
        current_source_functions=source_functions(current_source),
        semantics=dict(
            separator_topology="Old B2 line modes and occupied density nominate two distinct separator faces; this is absent from the B2-only V2 handoff.",
            side_physical_modes="Old B5 surveys Raw3D/BEV modes beyond B2 anchors, including local height persistence, corner topology and strip ownership.",
            profile_observations="Old prepare_geometry provides measured profile breaks/faces from per-segment local plane references; V2 constructs its shared refiner with an empty observation list.",
            raw_bev_dual_support="Old post-B2 side modes carry primary same-vessel Raw3D plus BEV support, separately from contextual lines.",
            partial_context="Old path retains physically supported candidates with explicit unresolved ownership and partial status.",
            boundary_ownership="Old profile observations are filtered by B1 vessel occupancy; V2 must retain its stricter selected-vessel isolation when adapting evidence.",
            independent_seed_path="Old B4 also consumes R2F/R2G physical rectangles, so an old final object is not identical to a same-numbered B2 proposal.",
        ),
        scenes=records,
        prohibited_shortcut="Do not execute prepare_geometry or a second global pipeline in the production adapter.",
        steel_edge_certified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-evidence-dir", type=Path, required=True)
    parser.add_argument("--before-funnel", type=Path, required=True)
    parser.add_argument("--old-source", type=Path, required=True)
    parser.add_argument("--current-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.old_evidence_dir,
        json.loads(args.before_funnel.read_bytes()),
        args.old_source,
        args.current_source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
