"""Offline engineering gate aggregation. This is never a control authorization."""

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = (
    "human_2d",
    "rigid_equivariance",
    "physical_ambiguity",
    "performance",
    "real11_repeatability",
    "sparse44",
    "physical_negatives",
    "golden12",
    "v14_b2_frozen",
    "windows",
    "ubuntu20",
    "ubuntu22",
    "same_sha_ci",
    "integration_safety",
    "rollback",
)


def evaluate(manifest, directory):
    """Missing, changed, empty, or differently versioned evidence fails closed."""
    sha = manifest["source_sha"]
    results = {}
    for name in REQUIRED:
        item = manifest.get("gates", {}).get(name, {})
        reasons = []
        if item.get("status") != "PASS":
            reasons.append(item.get("status", "NOT_VERIFIED"))
        if item.get("source_sha") != sha:
            reasons.append("SOURCE_SHA_MISMATCH")
        files = item.get("evidence", [])
        if not files:
            reasons.append("NO_EVIDENCE")
        for evidence in files:
            path = Path(directory) / evidence["path"]
            if not path.is_file():
                reasons.append("EVIDENCE_MISSING")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
                reasons.append("EVIDENCE_CHANGED")
        results[name] = dict(status="FAIL" if reasons else "PASS", reasons=reasons)
    return dict(
        source_sha=sha,
        decision=(
            "GO_OFFLINE_REVIEW_APPLICATION_ONLY"
            if all(r["status"] == "PASS" for r in results.values())
            else "NO_GO"
        ),
        gates=results,
        cross_vessel_generalization=manifest.get(
            "cross_vessel_generalization", "NOT_VERIFIED"
        ),
        device_control_allowed=False,
        steel_edge_certified=False,
        automatic_merge_allowed=False,
        limitation="Human signoff and evidence semantics require independent review; hashes establish provenance only.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.manifest.read_bytes()), args.manifest.parent)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(result["decision"])
    return 0 if result["decision"].startswith("GO_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
