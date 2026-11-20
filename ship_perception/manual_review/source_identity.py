"""Identify this worktree or a byte-verified offline release."""

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "mhr-build.json"
REQUIRED = (
    "ship_perception/manual_review/source_identity.py",
    "ship_perception/manual_review/workbench_v3.py",
    "ship_perception/manual_review/guided_v3.py",
    "ship_perception/manual_review/ui/workbench.js",
    "ship_perception/perception/pipeline.py",
    "tools/r3h_oneclick.py",
    "requirements-ros.txt",
)
SHA = re.compile("[a-f0-9]{40}")
HASH = re.compile("[a-f0-9]{64}")


def file_hash(path):
    with path.open("rb") as stream:
        h = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def verify_manifest(root, data):
    if (
        data["version"] != "MHR-R3"
        or data["protocols"] != [2, 3]
        or data["clean"] is not True
        or not SHA.fullmatch(data["build_sha"])
        or not HASH.fullmatch(data["production_source_sha256"])
    ):
        raise ValueError("MANIFEST_METADATA")
    files = data["files"]
    if not isinstance(files, dict) or not set(REQUIRED) <= files.keys():
        raise ValueError("MANIFEST_FILES")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or relative.as_posix() != name
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or not HASH.fullmatch(expected)
        ):
            raise ValueError("MANIFEST_PATH")
        path = root / name
        if not path.is_file() or path.resolve() != path or file_hash(path) != expected:
            raise ValueError("MANIFEST_HASH")
    relevant = {
        p.relative_to(root).as_posix()
        for folder in ("ship_perception", "tools")
        for p in (root / folder).rglob("*")
        if p.is_file()
        and p.suffix
        in {".py", ".json", ".yaml", ".yml", ".js", ".html", ".css", ".so", ".pyd"}
    }
    if relevant - files.keys():
        raise ValueError("MANIFEST_UNLISTED_SOURCE")


def identify(root=ROOT):
    root = root.resolve()
    packaged = root / MANIFEST
    try:
        if packaged.is_file():
            data = json.loads(packaged.read_bytes())
            verify_manifest(root, data)
            return dict(
                status="VERIFIED",
                build_sha=data["build_sha"],
                kind="RELEASE_MANIFEST",
                version=data["version"],
                clean=True,
                protocols=data["protocols"],
                manifest_sha256=file_hash(packaged),
                production_source_sha256=data["production_source_sha256"],
            )
    except (OSError, ValueError, LookupError, TypeError, AttributeError):
        return dict(
            status="UNVERIFIED", reason="INVALID_RELEASE_MANIFEST", build_sha=None
        )
    if not (root / ".git").exists():
        return dict(
            status="UNVERIFIED", reason="MISSING_PROJECT_IDENTITY", build_sha=None
        )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=5
        )

    try:
        if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
            raise ValueError("NOT_PROJECT_ROOT")
        git("ls-files", "--error-unmatch", "--", *REQUIRED)
        commit = git("rev-parse", "HEAD").decode().strip()
        clean = not git("status", "--porcelain", "--untracked-files=all")
        if not clean or not SHA.fullmatch(commit):
            raise ValueError("DIRTY_OR_INVALID_GIT")
        return dict(
            status="VERIFIED",
            build_sha=commit,
            kind="PROJECT_WORKTREE",
            version="MHR-R3",
            clean=True,
            protocols=[2, 3],
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return dict(
            status="UNVERIFIED",
            reason="DIRTY_OR_INVALID_PROJECT_WORKTREE",
            build_sha=None,
        )
