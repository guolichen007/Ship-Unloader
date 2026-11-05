"""Validated recall snapshots using whitelisted data types and NumPy arrays."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from .candidate_union import CandidateProposal
from .context import (
    SceneEvidenceContext, VesselEvidence, BoundaryMeasurementContext,
    BoundarySourceSlice, freeze,
)
from .region_model import OpeningRegion, TopologySupport, EvidenceSource, BoundaryMeasurement
from .structural_family import StructuralBoundaryFamily
from ship_perception.r1_static.height_grid import R1HeightGrid

TYPES={c.__name__:c for c in (
    SceneEvidenceContext,VesselEvidence,BoundaryMeasurementContext,BoundarySourceSlice,
    OpeningRegion,TopologySupport,EvidenceSource,BoundaryMeasurement,
    StructuralBoundaryFamily,CandidateProposal,R1HeightGrid,
)}
CACHE_ATTRIBUTES=("grids","context_grids","providers","provider_build_count","grid_build_counts",
                  "opening_identities","local_enclosure_evidence","s4_topology_bridge",
                  "structural_boundary_families","structural_family_associations",
                  "_v21_side_lines")


class RecallSnapshot:
    def __init__(self,path,identity):
        self.path=Path(path) if path else None
        self.identity=identity
        self.audit=dict(status="DISABLED" if self.path is None else "MISS")
        self.value=None
        if self.path is not None and self.path.is_file():
            try:
                self.value=self._read()
                self.audit["status"]="HIT"
            except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as error:
                self.audit.update(status="REBUILD",reason=type(error).__name__)

    def _read(self):
        with zipfile.ZipFile(self.path) as archive:
            manifest=json.loads(archive.read("manifest.json"))
            if manifest["identity"]!=self.identity:
                raise ValueError("SNAPSHOT_IDENTITY_MISMATCH")
            data=archive.read("state.json")
            if hashlib.sha256(data).hexdigest()!=manifest["state_sha256"]:
                raise ValueError("SNAPSHOT_CONTENT_MISMATCH")
            def decode(value):
                if isinstance(value,list):
                    return [decode(v) for v in value]
                if not isinstance(value,dict):
                    return value
                kind=value.get("__snapshot_kind__")
                if kind=="array":
                    raw=archive.read(value["name"])
                    if hashlib.sha256(raw).hexdigest()!=value["sha256"]:
                        raise ValueError("SNAPSHOT_ARRAY_MISMATCH")
                    array=np.load(io.BytesIO(raw),allow_pickle=False)
                    if array.dtype.hasobject:
                        raise ValueError("SNAPSHOT_OBJECT_ARRAY")
                    array.setflags(write=False)
                    return array
                if kind=="tuple":
                    return tuple(decode(v) for v in value["items"])
                if kind=="mapping":
                    result={decode(k):decode(v) for k,v in value["items"]}
                    return freeze(result) if value["frozen"] else result
                if kind=="dataclass":
                    cls=TYPES[value["type"]]
                    return cls(**{k:decode(v) for k,v in value["fields"].items()})
                if kind=="float":
                    return float(value["value"])
                raise ValueError("UNKNOWN_SNAPSHOT_RECORD")
            return decode(json.loads(data))

    def save(self,value):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix=self.path.name+".",suffix=".tmp",dir=str(self.path.parent))
        os.close(fd)
        try:
            with zipfile.ZipFile(name,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
                count=[0]
                def encode(item):
                    if isinstance(item,np.ndarray):
                        if item.dtype.hasobject:
                            raise ValueError("SNAPSHOT_OBJECT_ARRAY")
                        stream=io.BytesIO()
                        np.save(stream,item,allow_pickle=False)
                        raw=stream.getvalue()
                        key="arrays/"+str(count[0])+".npy"
                        count[0]+=1
                        archive.writestr(key,raw)
                        return dict(__snapshot_kind__="array",name=key,sha256=hashlib.sha256(raw).hexdigest())
                    if isinstance(item,np.generic):
                        return encode(item.item())
                    if is_dataclass(item):
                        if type(item).__name__ not in TYPES:
                            raise ValueError("SNAPSHOT_TYPE_NOT_ALLOWED:"+type(item).__name__)
                        return dict(__snapshot_kind__="dataclass",type=type(item).__name__,
                                    fields={f.name:encode(getattr(item,f.name)) for f in fields(item) if f.init})
                    if isinstance(item,Mapping):
                        return dict(__snapshot_kind__="mapping",frozen=type(item).__name__=="mappingproxy",
                                    items=[[encode(k),encode(v)] for k,v in item.items()])
                    if isinstance(item,tuple):
                        return dict(__snapshot_kind__="tuple",items=[encode(v) for v in item])
                    if isinstance(item,list):
                        return [encode(v) for v in item]
                    if isinstance(item,float) and not np.isfinite(item):
                        return dict(__snapshot_kind__="float",value=str(item))
                    if item is None or isinstance(item,(str,int,float,bool)):
                        return item
                    raise TypeError("SNAPSHOT_TYPE_NOT_ALLOWED:"+type(item).__name__)
                data=json.dumps(encode(value),allow_nan=False,separators=(",",":")).encode()
                archive.writestr("state.json",data)
                archive.writestr("manifest.json",json.dumps(dict(identity=self.identity,
                    state_sha256=hashlib.sha256(data).hexdigest(),array_count=count[0])))
            os.replace(name,self.path)
            self.audit.update(saved=True)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @staticmethod
    def cache_state(cache):
        return {key:getattr(cache,key) for key in CACHE_ATTRIBUTES if hasattr(cache,key)}

    @staticmethod
    def restore_cache(cache,state):
        for name,value in state.items():
            if name not in CACHE_ATTRIBUTES:
                raise ValueError("SNAPSHOT_CACHE_ATTRIBUTE_NOT_ALLOWED")
            setattr(cache,name,value)
