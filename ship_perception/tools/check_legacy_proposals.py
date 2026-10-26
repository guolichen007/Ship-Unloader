"""Compare complete frozen proposal/evidence hashes against private baselines."""

import argparse
import json
from pathlib import Path
from ship_perception.io.pcd_source import read_pcd
from ship_perception.perception.context import SceneEvidenceContext
from ship_perception.perception.local_evidence import VesselLocalEvidence
from ship_perception.perception.legacy_expert import (
    LegacyProposalExpert,
    verify_legacy_sources,
)
from ship_perception.perception.pipeline import software_identity, source_identity
from ship_perception.r1_static.run import resolve_config
from ship_perception.r1_static.heightmap_batch import RESEARCH_CONFIG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected = json.loads(args.baseline.read_text(encoding="utf-8"))
    frame = read_pcd(args.pcd)
    config, sha = resolve_config()
    research = json.loads(RESEARCH_CONFIG.read_bytes())
    verify_legacy_sources()
    if frame.sha256 != expected["semantic_sha256"]:
        raise ValueError("LEGACY_REGRESSION_INPUT_MISMATCH")
    scene = SceneEvidenceContext.build(
        frame, config, research, sha, software_identity()
    )
    results = []
    for vessel in scene.vessels:
        if (
            len(vessel.point_indexes)
            < research["min_points_per_coarse_cell"]
            * config["roi"]["min_deck_support_cells"]
        ):
            continue
        cache = VesselLocalEvidence.build(
            frame, vessel, config, scene.boundary_context(vessel.vessel_id)
        )
        batch = LegacyProposalExpert().propose(cache, research)
        reference = next(
            (v for v in expected["vessels"] if v["vessel_id"] == vessel.vessel_id), None
        )
        results.append(
            dict(
                vessel_id=vessel.vessel_id,
                proposal_count=len(batch.proposals),
                proposal_sha256=batch.output_sha256,
                expected_sha256=reference["proposal_sha256"] if reference else None,
                status=(
                    "PASS"
                    if reference and reference["proposal_sha256"] == batch.output_sha256
                    else "FAIL"
                ),
            )
        )
    passed = len(results) == len(expected["vessels"]) and all(
        r["status"] == "PASS" for r in results
    )
    report = dict(
        status="PASS" if passed else "FAIL",
        software_sha=software_identity(),
        source_identity=source_identity(),
        input_sha256=frame.provenance["wire_sha256"],
        vessels=results,
        scope="PROPOSAL_AND_EVIDENCE_ONLY_NOT_STEEL_GOLDEN",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
