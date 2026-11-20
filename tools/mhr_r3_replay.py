"""Offline fixed-draft replay. Archived selections are explicit test inputs only."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

from ship_perception.manual_review.contract import IDENTITY, canonical, selection
from ship_perception.manual_review.frame_session import FrameSession
from ship_perception.manual_review.guided_v3 import guided_match
from ship_perception.manual_review.workbench import SELECT
from ship_perception.manual_review.workbench_auto import load_auto


def read(path):
    return json.loads(path.read_bytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", type=Path)
    p.add_argument("requests", type=Path)
    p.add_argument("native", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--method", choices=("A", "B"), default="A")
    p.add_argument("--first-only", action="store_true")
    args = p.parse_args()
    options = {}
    if args.method == "B":
        from ship_perception.manual_review.physical_profile import augmented_profile

        options["primitive"] = augmented_profile
    specs = read(args.spec)
    native = {
        r["wire"]: next(
            (args.native / f'{r["case"]:02}').rglob("review_hatches_compact.json")
        ).parent
        for r in read(args.native / "receipt.json")
    }
    summary = []
    args.output.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        scene = spec["id"]
        started = perf_counter()
        frame = FrameSession(Path(spec["pcd"]))
        wire = frame.manifest["wire_sha256"]
        bundle = load_auto(native[wire], frame)
        cold_s = perf_counter() - started
        archived = read(args.requests / scene / "manifest.json")
        assert archived["wire_sha256"] == wire
        assert archived["semantic_sha256"] == frame.manifest["semantic_sha256"]
        folder = args.output / scene
        folder.mkdir(exist_ok=True)
        for case in sorted((args.requests / scene).glob("[0-9]-*.json")):
            if args.first_only and not case.stem.endswith("-first"):
                continue
            row = read(case)
            old = row["result"]
            request = {**row["request"], **{k: frame.manifest[k] for k in IDENTITY}}
            if request["bundle_digest"] is None:
                targets = {}
            else:
                assert request["bundle_digest"] == bundle["bundle_digest"]
                targets = bundle["targets"]
            xy = selection(
                {k: request[k] for k in set(IDENTITY) | SELECT},
                frame.manifest,
                frame.projection,
                targets,
            )
            result = guided_match(frame, xy, request, targets, **options)
            report = dict(
                scene=scene,
                case=case.stem,
                wire=wire,
                baseline_reason=old.get("reason"),
                operation=request["operation"],
                state=result["match_state"],
                reason=result["reason"],
                search_complete=result["search_complete"],
                cold_s=cold_s,
                phase_s=result["phases_s"],
                seconds=result["elapsed_s"],
                candidates=[
                    dict(
                        id=c["candidate_id"],
                        confirmable=c["confirmable"],
                        z_null=c["geometry"]["z"].count(None),
                    )
                    for c in result["candidates"]
                ],
                side_counts=[s["mode_count"] for s in result["sides"]],
                training_feasible=result.get("training_feasible", 0),
            )
            (folder / case.name).write_bytes(
                canonical(dict(report=report, request=request, result=result))
            )
            summary.append(report)
        print(
            scene,
            Counter((r["state"], r["reason"]) for r in summary if r["scene"] == scene),
            flush=True,
        )
    (args.output / "summary.json").write_bytes(canonical(summary))
    print(Counter((r["state"], r["reason"]) for r in summary), flush=True)


if __name__ == "__main__":
    main()
