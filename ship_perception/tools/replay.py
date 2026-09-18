"""Export a replay with an input SHA-256 and UTC provenance manifest."""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--executable", required=True)
    p.add_argument("--input", help="PCD with LOCAL XYZ, never absolute float world XYZ")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    data = pathlib.Path(args.input).resolve() if args.input else None
    if data and not data.is_file():
        p.error("input PCD does not exist")
    digest = hashlib.sha256(data.read_bytes()).hexdigest() if data else None
    subprocess.run([str(pathlib.Path(args.executable).resolve()), str(data) if data else "--synthetic",
                    args.output], check=True)
    path = pathlib.Path(args.output) / "metadata.json"
    manifest = json.loads(path.read_text())
    manifest.update(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    dataset_id=("pcd_sha256:" + digest) if data else "synthetic_ship_grid_v1",
                    input_sha256=digest, input_path=str(data) if data else None)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
