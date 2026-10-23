"""One scene discovery and immutable, hashed point ownership per input frame."""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from ship_perception.r1_static.scene_vessel import infer_scene_vessels


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if hasattr(value, "items"):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class VesselEvidence:
    public: object
    point_indexes: np.ndarray
    point_index_sha256: str

    @property
    def vessel_id(self):
        return self.public["vessel_hypothesis_id"]

    @property
    def axes(self):
        result = np.array(self.public["local_axes"], dtype=float)
        result.setflags(write=False)
        return result

    def identity(self):
        return dict(
            vessel_id=self.vessel_id,
            point_count=len(self.point_indexes),
            point_index_sha256=self.point_index_sha256,
            bbox=thaw(self.public["bbox_xy"]),
            local_axes=self.axes.tolist(),
            support_ids=thaw(self.public["source_support_ids"]),
            classification_status=self.public["classification_status"],
        )


@dataclass(frozen=True)
class SceneEvidenceContext:
    input_semantic_sha256: str
    raw_identity_json: str
    config_sha256: str
    software_sha: str
    scene: object
    vessels: tuple

    @classmethod
    def build(cls, frame, config, research, config_sha, software_sha):
        scene, private = infer_scene_vessels(frame.xyz, config, research)
        vessels = []
        for row in scene["vessel_hypotheses"]:
            ownership_bytes = np.asarray(
                private["vessel_point_indexes"][row["vessel_hypothesis_id"]], dtype="<i8"
            ).tobytes()
            # Immutable bytes also prohibit re-enabling ndarray writeability.
            indexes = np.frombuffer(ownership_bytes, dtype="<i8")
            vessels.append(
                VesselEvidence(
                    freeze(row),
                    indexes,
                    hashlib.sha256(memoryview(indexes).cast("B")).hexdigest(),
                )
            )
        return cls(
            frame.sha256,
            frame.provenance_json,
            config_sha,
            software_sha,
            freeze(scene),
            tuple(vessels),
        )

    def validate_frame(self, frame):
        if frame.sha256 != self.input_semantic_sha256:
            raise ValueError("SCENE_CONTEXT_INPUT_IDENTITY_MISMATCH")
        if frame.provenance_json != self.raw_identity_json:
            raise ValueError("SCENE_CONTEXT_WIRE_IDENTITY_MISMATCH")

    def audit(self):
        return dict(
            input_semantic_sha256=self.input_semantic_sha256,
            raw_identity=json.loads(self.raw_identity_json),
            config_sha256=self.config_sha256,
            software_sha=self.software_sha,
            b1_execution_count=1,
            vessels=[v.identity() for v in self.vessels],
        )
