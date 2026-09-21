"""PCD 字段偏移与拒绝路径测试；不依赖真实 holdout。"""
from pathlib import Path
import struct
import hashlib
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from v15_pcd import decode, digest, UnsupportedPCD


def header(fields="x y z", sizes="4 4 4", types="F F F", counts="1 1 1", data="binary", points=2):
    return (f"VERSION .7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n"
            f"WIDTH {points}\nHEIGHT 1\nVIEWPOINT 5 6 7 1 0 0 0\nPOINTS {points}\nDATA {data}\n").encode("ascii")


class DecoderTests(unittest.TestCase):
    def decode_bytes(self, raw, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.pcd"
            path.write_bytes(raw)
            return decode(path, **kwargs)

    def test_layout_and_viewpoint(self):
        raw=header("_ z x _ y", "1 4 4 1 4", "U F F U F", "4 1 1 4 1")
        raw+=b"abcd"+struct.pack("<ff",3,1)+b"efgh"+struct.pack("<f",2)
        raw+=b"ABCD"+struct.pack("<ff",6,4)+b"EFGH"+struct.pack("<f",5)
        xyz,_=self.decode_bytes(raw)
        np.testing.assert_array_equal(xyz,[[1,2,3],[4,5,6]])

    def test_audited_zero_tail(self):
        text=header()
        payload=struct.pack("<ffffff",1,2,3,4,5,6)
        raw=text+payload+b"\0"*(4096-len(text))
        contract=[dict(expected_file_sha256=hashlib.sha256(raw).hexdigest(),
                       expected_point_payload_bytes=len(payload),expected_trailing_zero_bytes=4096-len(text))]
        with self.assertRaises(UnsupportedPCD):self.decode_bytes(raw)
        xyz,meta=self.decode_bytes(raw,padding_contracts=contract)
        self.assertEqual(len(xyz),2)
        self.assertIn("AUDITED_ZERO_TAIL_BYTES",meta)
        changed=text+struct.pack("<ffffff",9,2,3,4,5,6)+b"\0"*(4096-len(text))
        with self.assertRaises(UnsupportedPCD):self.decode_bytes(changed,padding_contracts=contract)
        for field in ("expected_point_payload_bytes","expected_trailing_zero_bytes"):
            wrong=[dict(contract[0])];wrong[0][field]+=1
            with self.assertRaises(UnsupportedPCD):self.decode_bytes(raw,padding_contracts=wrong)
        for suffix in (b"\0",b"x"*(4096-len(text)),b"\0"*(4095-len(text))):
            with self.assertRaises(UnsupportedPCD):self.decode_bytes(text+payload+suffix)

    def test_rejects_unknown_and_corrupt_layouts(self):
        payload=struct.pack("<ffffff",1,2,3,4,5,6)
        invalid=[header(data="binary_compressed"),header(data="ascii"),header(counts="1 0 1"),
                 header(fields="x x z"),header(fields="x intensity z"),header(sizes="8 4 4"),
                 header(types="I F F"),header(points=3),header().replace(b"WIDTH 2",b"WIDTH 3")]
        for text in invalid:
            with self.subTest(text=text),self.assertRaises(UnsupportedPCD):self.decode_bytes(text+payload)
        with self.assertRaises(UnsupportedPCD):self.decode_bytes(header()+payload[:-1])
        for value in (float("nan"),float("inf"),float("-inf")):
            with self.assertRaises(UnsupportedPCD):self.decode_bytes(header()+struct.pack("<ffffff",value,2,3,4,5,6))

    def test_canonical_hash_retains_multiplicity(self):
        a=np.array([[0.,1,2],[3,4,5],[3,4,5]],dtype="<f4")
        b=a[::-1].copy();b[-1,0]=-0.
        self.assertEqual(digest(a)["unordered_xyz_sha256"],digest(b)["unordered_xyz_sha256"])
        self.assertNotEqual(digest(a)["ordered_xyz_sha256"],digest(b)["ordered_xyz_sha256"])
        self.assertNotEqual(digest(a)["unordered_xyz_sha256"],digest(a[:2])["unordered_xyz_sha256"])


if __name__=="__main__":unittest.main()
