"""Build offline MHR packages from a clean repository and verify their manifests."""

import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from ship_perception.manual_review.contract import canonical
from ship_perception.manual_review.source_identity import identify
from ship_perception.perception.pipeline import source_identity

SDK = (
    "ship_perception/manual_review/ui/workbench.js",
    "ship_perception/manual_review/ui/workbench.d.ts",
    "ship_perception/manual_review/ui/style.css",
    "ship_perception/manual_review/ui/bootstrap.js",
    "ship_perception/manual_review/ui/example.html",
    "docs/MHR-R3接口.md",
    "docs/MHR-R3-openapi.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    provenance = identify(root)
    if provenance["status"] != "VERIFIED" or provenance["kind"] != "PROJECT_WORKTREE":
        parser.error("Run from a clean MHR project worktree.")
    sha = provenance["build_sha"]
    archive = BytesIO(subprocess.check_output(["git", "archive", "--format=zip", sha]))
    args.output.mkdir(parents=True, exist_ok=True)
    offline = args.output / "ShipUnloader_MHR_R3_Run.zip"
    sdk = args.output / "ShipUnloader_MHR_R3_SDK.zip"
    with ZipFile(archive) as original:
        names = [n for n in original.namelist() if not n.endswith("/")]
        for n in names:
            if n.startswith(("Ship-Unloader-Data/", "Ship-Unloader-Work/")):
                raise ValueError("PRIVATE_ARTIFACT")
            if (
                n.lower().endswith((".pcd", ".ply", ".bag", ".npz"))
                and n != "ship_perception/tests/fixtures/local_xyz.pcd"
            ):
                raise ValueError("UNEXPECTED_POINT_CLOUD")
        software = dict(software_git_sha=sha)
        manifest = dict(
            version="MHR-R3",
            protocols=[2, 3],
            clean=True,
            build_sha=sha,
            production_source_sha256=source_identity()["runtime_source_sha256"],
            dependencies=original.read("requirements-ros.txt").decode().splitlines(),
            files={},
        )
        with ZipFile(offline, "w", ZIP_DEFLATED) as z:
            for name in names:
                body = original.read(name)
                manifest["files"][name] = sha256(body).hexdigest()
                z.writestr(name, body)
            meta = "ship_perception/software_identity.json"
            metadata = canonical(software)
            z.writestr(meta, metadata)
            manifest["files"][meta] = sha256(metadata).hexdigest()
            z.writestr("mhr-build.json", canonical(manifest))
        with ZipFile(sdk, "w", ZIP_DEFLATED) as z:
            contents = {Path(name).name: original.read(name) for name in SDK}
            for name, body in contents.items():
                z.writestr(name, body)
            z.writestr(
                "mhr-sdk-build.json",
                canonical(
                    dict(
                        version="MHR-R3",
                        protocols=[2, 3],
                        build_sha=sha,
                        files={n: sha256(b).hexdigest() for n, b in contents.items()},
                    )
                ),
            )
    with TemporaryDirectory(dir=args.output) as temp, ZipFile(offline) as z:
        z.extractall(temp)
        assert identify(Path(temp))["build_sha"] == sha
    for path in (sdk, offline):
        print(path, sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
