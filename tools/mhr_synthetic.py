"""Offline PCD fixture generator. No fixture truth is imported by the runtime."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ship_perception.perception.measurement import RawSpatialIndex


def rect(c, theta, length, width):
    e = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-e[1], e[0]])
    return c + np.array(
        [
            a * e * length / 2 + b * n * width / 2
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )


def synthesized(xy, variant="clean", rng=None):
    rows = []
    for i, a in enumerate(xy):
        if (variant == "occluded" and i == 0) or variant == "all_occluded":
            continue
        b = xy[(i + 1) % 4]
        length = np.linalg.norm(b - a)
        e = (b - a) / length
        n = np.array([-e[1], e[0]])
        stations = np.arange(0.09, length - 0.08, 0.095)
        if variant == "cluster":
            stations = stations[np.abs(stations - length / 2) < 0.6]
        if variant == "sparse":
            stations = stations[np.abs(stations - length / 2) < 1.1]
        for offset, z in [(0, v) for v in (-0.85, -0.68, -0.51, -0.34, -0.17, 0)] + [
            (-0.19, 0),
            (-0.36, 0),
            (-0.52, 0),
            (0.19, -1),
            (0.36, -1),
            (0.54, -1),
            (-0.93, -0.72),
        ]:
            if variant == "high_cargo" and offset > 0:
                z = 0.50
            if variant == "shore" and i > 0:
                z = -0.6
            pts = a + stations[:, None] * e + offset * n
            rows.append(np.column_stack([pts, np.full(len(pts), z + 3.2)]))
        if variant in ("dual", "twin") and i == 0:
            # Separate outward-top strip with a matching inward depression.
            for offset, z in [
                (1.3, v) for v in (-0.85, -0.68, -0.51, -0.34, -0.17, 0)
            ] + [(1.11, 0), (0.94, 0), (0.78, 0), (1.49, -1), (1.66, -1), (1.84, -1)]:
                pts = a + stations[:, None] * e + offset * n
                rows.append(np.column_stack([pts, np.full(len(pts), z + 3.2)]))
    if not rows:
        return np.column_stack([xy, np.zeros(len(xy))])
    p = np.vstack(rows)
    if variant == "noisy":
        p += rng.normal(0, 0.004, p.shape)
        p = p[rng.random(len(p)) > 0.12]
    if variant == "cropped":
        p = p[p[:, 0] < xy[:, 0].mean()]
    if variant == "deformed":
        p[:, 0] += 0.13 * (p[:, 1] - xy[:, 1].mean())
    return p


def frame_for(xyz):
    return SimpleNamespace(frame=SimpleNamespace(xyz=xyz), index=RawSpatialIndex(xyz))


def save_pcd(path, xyz):
    with Path(path).open("wb") as stream:
        stream.write(
            (
                "VERSION .7\nFIELDS x y z\nSIZE 8 8 8\nTYPE F F F\n"
                f"WIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA binary\n"
            ).encode()
        )
        stream.write(xyz.astype("<f8").tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    xy = rect(np.array([12.0, -22.0]), 0, 18.0, 9.0)
    boundary = rect(np.array([12.0, -22.0]), 0, 24.0, 15.0)
    save_pcd(
        args.output,
        np.vstack((synthesized(xy), np.column_stack((boundary, np.zeros(4))))),
    )


if __name__ == "__main__":
    main()
