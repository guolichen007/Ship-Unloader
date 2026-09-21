"""跨平台配置身份；CRLF/LF、缩进与 key 顺序不改变参数身份。"""
import hashlib
import json
from pathlib import Path


def canonical_bytes(data):
    if isinstance(data,(bytes,str)):data=json.loads(data)
    return json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def config_hash(data):
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def config_file_hash(path):
    return config_hash(Path(path).read_bytes())
