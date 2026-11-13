"""Profile the existing offline PCD review without changing its inputs or limits."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import cProfile
from functools import wraps
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from unittest.mock import patch

import run_opening_ir_shadow
from ship_perception.perception import measurement, opening_rim_source, review_height
from ship_perception.perception import review_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = defaultdict(lambda: dict(calls=0, wall_s=0.0, cpu_s=0.0))
    active = []
    lock = threading.Lock()
    stop = threading.Event()
    main_thread = threading.get_ident()
    profiler = cProfile.Profile()
    profile_active = False

    def timed(owner, name):
        original = getattr(owner, name)
        label = f"{owner.__name__}.{name}"

        @wraps(original)
        def call(*a, **kw):
            nonlocal profile_active
            top = owner is review_height.ReviewCornerHeight and name == "measure"
            start = time.perf_counter()
            cpu = time.process_time()
            with lock:
                active.append(label)
            if top:
                profiler.enable()
                profile_active = True
            try:
                return original(*a, **kw)
            finally:
                if top:
                    profiler.disable()
                    profile_active = False
                    profiler.dump_stats(str(args.output_dir / "corner_z.pstats"))
                with lock:
                    active.pop()
                    row = records[label]
                    row["calls"] += 1
                    row["wall_s"] += time.perf_counter() - start
                    row["cpu_s"] += time.process_time() - cpu

        return patch.object(owner, name, call)

    def sample(stream):
        frame = sys._current_frames().get(main_thread)
        with lock:
            row = dict(
                t_monotonic=time.monotonic(),
                active_calls=list(active),
                corner_z_profile_active=profile_active,
                completed={k: dict(v) for k, v in records.items()},
                stack=traceback.format_stack(frame, limit=18) if frame else [],
            )
        stream.write(json.dumps(row) + "\n")
        stream.flush()

    def sample_until_done(stream):
        while not stop.wait(10):
            sample(stream)

    with ExitStack() as stack:
        targets = {
            review_height.ReviewCornerHeight: (
                "measure", "_plane_candidates", "_support_points", "_normal_support",
            ),
            opening_rim_source.OpeningRimSource: (
                "collect", "_patches", "_components", "_low_attachment", "positions",
            ),
            review_height: ("_gravity_hypotheses", "review_normal_cache"),
            measurement.BudgetedLocalSurfaceNormals: ("_estimate",),
            review_runtime: ("_atomic",),
        }
        for owner, names in targets.items():
            for name in names:
                stack.enter_context(timed(owner, name))
        stream = stack.enter_context((args.output_dir / "profile_samples.jsonl").open("w"))
        sampler = threading.Thread(target=sample_until_done, args=(stream,), daemon=True)
        sampler.start()
        original_argv = sys.argv
        sys.argv = [original_argv[0], "--pcd", str(args.pcd),
                    "--output-dir", str(args.output_dir)]
        try:
            return run_opening_ir_shadow.main()
        finally:
            sys.argv = original_argv
            stop.set()
            sampler.join()
            sample(stream)


if __name__ == "__main__":
    raise SystemExit(main())
