"""Capture reproducible source changes without modifying Git metadata."""

import difflib
import hashlib
from pathlib import Path
import subprocess


def git_source_state(root):
    root = Path(root)
    if not (root / ".git").exists():
        return (
            dict(git_head_sha=None, git_worktree_dirty=None, git_diff_sha256=None),
            None,
        )

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=str(root), stderr=subprocess.DEVNULL
        )

    patch = git("diff", "--binary", "--no-ext-diff", "--no-color", "HEAD", "--")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for raw_name in sorted(name for name in untracked if name):
        name = raw_name.decode("utf-8")
        path = root / name
        # Work/data/output are ignored. Include new runtime/config/test sources
        # in the patch even before git add; a tracked-only diff is incomplete.
        if path.suffix.lower() not in (
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".sh",
            ".md",
            ".txt",
            ".cmake",
        ):
            continue
        contents = path.read_text(encoding="utf-8").splitlines(keepends=True)
        diff = "diff --git a/{0} b/{0}\nnew file mode 100644\n".format(name)
        diff += "".join(
            difflib.unified_diff([], contents, fromfile="/dev/null", tofile="b/" + name)
        )
        if contents and not contents[-1].endswith("\n"):
            diff += "\n\\ No newline at end of file\n"
        patch += diff.encode("utf-8")
    # Compare actual canonical content rather than Git's unrefreshed stat
    # cache (read-only execution cannot update the index after line endings).
    dirty = bool(patch) or bool(git("ls-files", "--others", "--exclude-standard", "-z"))
    return dict(
        git_head_sha=git("rev-parse", "HEAD").decode().strip(),
        git_worktree_dirty=dirty,
        git_diff_sha256=hashlib.sha256(patch).hexdigest(),
    ), patch.decode("utf-8")
