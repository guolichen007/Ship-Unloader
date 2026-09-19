"""从不可变 BASE_SHA 校验冻结路径；不能用分支内清单重新定义基线。"""
import hashlib
import json
import subprocess
from pathlib import Path

BASE_SHA = "d586b93fbf676e585e77286b7b10d7076a64d3b7"
ROOT = Path(__file__).resolve().parents[2]
PATHS = ["ship_perception/src/registration", "ship_perception/src/tracking",
         "ship_perception/src/mapping", "ship_perception/include/ship_perception/registration",
         "ship_perception/include/ship_perception/tracking", "ship_perception/include/ship_perception/mapping",
         "ship_perception/config/v14.json", "ship_perception/config/v14.schema.json",
         "ship_perception/tools/configure_v14.py", "ship_perception/tools/run_v14.py",
         "ship_perception/tests/v14_fixture.hpp", "ship_perception/tests/v14_evaluate.cpp",
         "ship_perception/tests/v14_safety.cpp", "ship_perception/tests/v14_backend.cpp",
         "ship_perception/tests/test_v14_tools.py", "ship_perception/cmake/SmallGicp.cmake",
         "ship_perception/tools/verify_vendor.py", "third_party/small_gicp"]


def verify():
    raw = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE_SHA, "--", *PATHS], cwd=ROOT)
    names = raw.decode().splitlines()
    changed = []
    for name in names:
        expected = subprocess.check_output(["git", "show", BASE_SHA+":"+name], cwd=ROOT)
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(expected).digest():
            changed.append(name)
    # additions inside frozen directories also change the frozen implementation
    for part in PATHS:
        p = ROOT/part
        if p.is_dir():
            changed.extend(f.relative_to(ROOT).as_posix() for f in p.rglob("*")
                           if f.is_file() and f.relative_to(ROOT).as_posix() not in names)
    if changed:
        raise ValueError("V14_FROZEN_PATH_FAIL: "+repr(changed))
    return dict(status="PASS", base_sha=BASE_SHA, checked_files=len(names))


if __name__ == "__main__":
    print(json.dumps(verify()))
