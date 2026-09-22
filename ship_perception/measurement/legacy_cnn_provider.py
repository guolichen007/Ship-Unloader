"""Read-only legacy CNN provider for R0 measurement, launched as a separate process.

This is not a product adapter. It preserves the old absolute-Y search and model
weights deliberately, so their failure can be measured without silently fixing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

from ship_perception.measurement.proposal import Candidate, nms
from ship_perception.tools.v15_pcd import decode


def run(pcd: Path, legacy_dir: Path, *, batch_size=64):
    sys.path.insert(0, str(legacy_dir))
    import train  # legacy read-only independent provider; never linked to production
    from analyze_pcd import build_grid
    from export_dataset import crop_resize

    model_path = legacy_dir / "cnn_model.bin"
    raw = model_path.read_bytes()
    if raw[:8] != b"HATCHCNN" or struct.unpack("<I", raw[8:12])[0] != 8:
        raise ValueError("UNSUPPORTED_LEGACY_MODEL_FORMAT_OR_NORMALIZATION")
    points, _ = decode(pcd)
    surf, _, valid, meta = build_grid(points, 0.5, 5, 0.5)
    xbase, ybase, nx, ny, cs = meta
    params = train.load_model(str(model_path))
    batches = []
    rectangles = []
    all_candidates = []
    counts = {"window_enumerated": 0, "occupancy_rejected": 0,
              "contrast_rejected": 0, "crop_rejected": 0, "cnn_scored": 0,
              "below_score_threshold": 0}
    z = np.where(valid, surf, 0.0)
    isum = np.pad(z, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    icnt = np.pad(valid.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def rect_stats(r0, r1, c0, c1):
        r0, r1 = max(0, r0), min(ny, r1)
        c0, c1 = max(0, c0), min(nx, c1)
        if r1 <= r0 or c1 <= c0:
            return float("nan"), 0.0
        n = icnt[r1, c1]-icnt[r0, c1]-icnt[r1, c0]+icnt[r0, c0]
        s = isum[r1, c1]-isum[r0, c1]-isum[r1, c0]+isum[r0, c0]
        return (float(s/n), float(n)) if n else (float("nan"), 0.0)

    def flush():
        if not batches:
            return
        images = np.stack(batches)[:, None]
        logits, _ = train.cnn_forward(images, params)
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -30, 30)))
        for rect, score in zip(rectangles, scores):
            counts["cnn_scored"] += 1
            if score <= 0.5:
                counts["below_score_threshold"] += 1
                continue
            all_candidates.append(Candidate("LEGACY_CNN", f"cnn-{len(all_candidates):07d}",
                                            float(score), tuple(float(v) for v in rect),
                                            "legacy_model_baseline"))
        batches.clear()
        rectangles.clear()

    # The ranges, strict inequalities and gate replicate the C++ collectCandidates
    # defaults, including its known absolute-Y prior and fixed window dimensions.
    for y0 in np.arange(1.5, 7.5 + 1e-9, 0.5):
        for depth in np.arange(8.0, 25.0 + 1e-9, 1.0):
            y1 = y0 + depth
            if y1 > ybase + ny*cs:
                break
            rs0 = max(0, int((y0-2-ybase)/cs))
            rs1 = max(0, int((y0-ybase)/cs))
            rn0 = min(ny, int((y1-ybase)/cs))
            rn1 = min(ny, int((y1+2-ybase)/cs))
            ri0 = max(0, int((y0-ybase)/cs))
            ri1 = min(ny, int((y1-ybase)/cs))
            for x0 in np.arange(xbase, xbase+nx*cs-25+1e-9, 2.0):
                for x1 in np.arange(x0+25, min(xbase+nx*cs, x0+105)+1e-9, 2.0):
                    counts["window_enumerated"] += 1
                    ci0 = max(0, int((x0-xbase)/cs))
                    ci1 = min(nx, int((x1-xbase)/cs))
                    mi, ni = rect_stats(ri0, ri1, ci0, ci1)
                    if not np.isfinite(mi) or ni < 0.6*(ri1-ri0)*(ci1-ci0):
                        counts["occupancy_rejected"] += 1
                        continue
                    ms, ns = rect_stats(rs0, rs1, ci0, ci1)
                    mn, nn = rect_stats(rn0, rn1, ci0, ci1)
                    ring = ((ms*ns if ns else 0)+(mn*nn if nn else 0))/(ns+nn) if ns+nn else 0
                    if abs(mi-ring) < 0.35:
                        counts["contrast_rejected"] += 1
                        continue
                    rect = (x0, y0, x1, y1)
                    patch = crop_resize(surf, valid, meta, rect, 32, 16)
                    if patch is None:
                        counts["crop_rejected"] += 1
                        continue
                    batches.append(patch)
                    rectangles.append(rect)
                    if len(batches) == batch_size:
                        flush()
    flush()
    audited = nms(all_candidates, threshold=0.3, top_k=5)
    return {"provider": "LEGACY_CNN", "model_sha256": hashlib.sha256(raw).hexdigest(),
            "input_pcd_sha256": hashlib.sha256(pcd.read_bytes()).hexdigest(),
            "known_limitations": ["absolute_y_prior", "fixed_window_dimensions", "trained_on_partial_weak_labels"],
            "counts": counts, "candidates": [c.record() for c in audited]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pcd", type=Path, required=True)
    ap.add_argument("--legacy-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = run(args.pcd, args.legacy_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
