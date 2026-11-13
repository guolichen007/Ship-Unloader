"""Generate a gravity-preserving diagnostic PCD with traceable input rows."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ship_perception.io.pcd_source import read_pcd


def transform(points, yaw_deg, origin_xy, translation_xyz):
    angle = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    xyz = np.asarray(points, dtype=np.float64)
    xy = (xyz[:, :2] - origin_xy) @ rotation.T + origin_xy + translation_xyz[:2]
    return np.column_stack((xy, xyz[:, 2] + translation_xyz[2])), rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("f32", "f64"), required=True)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--origin-xy", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--translation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--shuffle-seed", type=int)
    args = parser.parse_args()
    frame = read_pcd(args.pcd)
    if frame.intensity is not None:
        raise ValueError("PLANAR_FIXTURE_REQUIRES_XYZ_ONLY")
    origin = np.asarray(args.origin_xy)
    translation = np.asarray(args.translation)
    points, rotation = transform(frame.xyz, args.yaw_deg, origin, translation)
    if not np.isfinite(points).all():
        raise ValueError("NONFINITE_PLANAR_TRANSFORM")
    rows = np.arange(len(points), dtype="<i8")
    if args.shuffle_seed is not None:
        np.random.default_rng(args.shuffle_seed).shuffle(rows)
    size = 4 if args.precision == "f32" else 8
    stored = points[rows].astype("<f" + str(size))
    if not np.isfinite(stored).all():
        raise ValueError("NONFINITE_STORED_PLANAR_TRANSFORM")
    args.output_dir.mkdir(parents=True)
    pcd = args.output_dir / "input.pcd"
    with pcd.open("wb") as stream:
        stream.write((
            "VERSION .7\nFIELDS x y z\nSIZE {0} {0} {0}\nTYPE F F F\n"
            "COUNT 1 1 1\nWIDTH {1}\nHEIGHT 1\nPOINTS {1}\nDATA binary\n"
        ).format(size, len(rows)).encode("ascii"))
        stream.write(stored.tobytes())
    np.save(args.output_dir / "source_finite_indexes.npy", rows)
    metadata = dict(
        original_input_sha256=frame.provenance["wire_sha256"],
        derived_input_sha256=hashlib.sha256(pcd.read_bytes()).hexdigest(),
        source_finite_indexes_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        original_source_point_count=frame.source_point_count,
        finite_point_count=len(rows),
        precision=args.precision, yaw_deg=args.yaw_deg, origin_xy=origin.tolist(),
        translation_xyz=translation.tolist(), rotation_xy=rotation.tolist(),
        shuffle_seed=args.shuffle_seed,
        max_store_component_error_m=float(np.max(np.abs(points[rows] - stored))),
        frame_contract="GRAVITY_PRESERVING_XYZ_WITHOUT_ACTIVE_BERTH_REFERENCE",
    )
    (args.output_dir / "transform.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
