"""Create an isolated, traceable source package; never include local PCD/Work."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args, root=ROOT):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def package(output, algorithm_base):
    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise ValueError("REVIEW_PACKAGE_REQUIRES_CLEAN_SOURCE")
    code_paths = [
        "ship_perception/perception",
        "ship_perception/io",
        "ship_perception/r1_static",
        "ship_perception/ros2",
        "ship_perception/config",
        "ship_perception/tools",
    ]
    if git("diff", "--name-only", algorithm_base, sha, "--", *code_paths):
        raise ValueError("ALGORITHM_BASE_CHANGED_DURING_REVIEW_PACKAGING")
    output = Path(output).resolve()
    target = output / ("ship_hatch_review_" + sha[:12])
    target.mkdir(parents=True, exist_ok=False)
    checkout = target / "Ship-Unloader"
    subprocess.run(
        [
            "git",
            "clone",
            "--config",
            "core.autocrlf=false",
            "--no-hardlinks",
            "--single-branch",
            str(ROOT),
            str(checkout),
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", sha], cwd=checkout, check=True)
    # Retain real Git objects for immutable freeze verification, but do not
    # ship a remote pointing to the developer's local filesystem.
    remote = git("remote", "get-url", "origin")
    if not remote.startswith("git@github.com:"):
        raise ValueError("REVIEW_PACKAGE_REQUIRES_PUBLIC_SSH_REMOTE")
    subprocess.run(
        ["git", "remote", "set-url", "origin", remote], cwd=checkout, check=True
    )
    launch = '#!/usr/bin/env bash\nset -euo pipefail\nROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\nexec bash "$ROOT/Ship-Unloader/scripts/run_ship_hatch_review.sh" "$@"\n'
    (target / "run_review.sh").write_bytes(launch.encode("utf-8"))
    guide = """# 船舱标定人工审查包

本包固定算法，只用于人工复核，未声明钢内沿、独立精度、速度或现场验收通过。

Ubuntu 22.04 已安装依赖后，一键运行：

```bash
bash run_review.sh --pcd /data/map.pcd --output-dir /data/review
bash run_review.sh --pcd-dir /data/pcd --output-dir /data/review
bash run_review.sh --manifest /data/review_batch.json --output-dir /data/review
```

首次依赖准备：

```bash
sudo apt-get install python3-venv fonts-noto-cjk
python3 -m venv Ship-Unloader/.local/review-venv
Ship-Unloader/.local/review-venv/bin/python -m pip install -r Ship-Unloader/requirements-ros.txt
```

离线 PCD 不需要 ROS；已有环境可用 PYTHON_BIN=/absolute/path/to/python 指定。
审查入口显式采用 review-first，不改变 detector 默认 bounded。结果目录包含 index.html、原始输入/派生裁剪独立矩阵、每场标定图和四角 XYZ JSON、原始日志、代码及输入身份。Z 缺测保留 null；超时及选船歧义不当作识别成功。按 Ctrl+C 会停止当前子进程并保留已完成产物。

多船只接受已有现场参考或操作员船体 ROI；ROI 不提供舱边界。清单不得包含历史坐标、Golden、期望舱数或命令。示例格式见 Ship-Unloader/docs/02_ROS2现场一键识别运行说明.md。
"""
    (target / "启动说明.md").write_text(guide, encoding="utf-8")
    files = {
        p.relative_to(checkout).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in checkout.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(checkout).parts
    }
    manifest = dict(
        schema="ship_perception.review_source_package.1",
        source_git_sha=sha,
        algorithm_base_sha=algorithm_base,
        algorithm_runtime_paths_unchanged=True,
        source_remote=remote,
        raw_customer_data_included=False,
        source_files=files,
    )
    (target / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def portable(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.isdir() or info.name.endswith(".sh") else 0o644
        return info

    archive = output / (target.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(target, arcname=target.name, filter=portable)
    receipt = dict(
        directory=str(target),
        archive=str(archive),
        source_git_sha=sha,
        algorithm_base_sha=algorithm_base,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    (output / "PACKAGE_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithm-base-sha", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package(args.output_dir, args.algorithm_base_sha),
            ensure_ascii=False,
            indent=2,
        )
    )
