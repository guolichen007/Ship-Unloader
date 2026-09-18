"""校验固定第三方源码；禁止配置阶段联网获取替代版本。"""
import hashlib
import json
import pathlib
import sys


def verify(directory):
    root = pathlib.Path(directory)
    expected = json.loads((root / "SOURCE_SHA256.json").read_text(encoding="utf-8"))
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("上游源码校验失败: " + name)
    imported = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if imported != set(expected) | {"SOURCE_SHA256.json", "UPSTREAM.md"}:
        raise ValueError("上游快照出现未登记文件")


if __name__ == "__main__":
    verify(sys.argv[1])
