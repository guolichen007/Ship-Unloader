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
class BoundarySourceSlice:
    origin: object
    point_indexes: np.ndarray
    point_index_sha256: str
    support_ids: tuple

    @property
    def vessel_id(self):
        return self.origin.vessel_id

    def identity(self):
        return dict(
            self.origin.identity(),
            point_count=len(self.point_indexes),
            point_index_sha256=self.point_index_sha256,
            support_ids=list(self.support_ids),
            origin_point_index_sha256=self.origin.point_index_sha256,
            origin_support_ids=thaw(self.origin.public["source_support_ids"]),
            attribution_method="UNIQUE_SUPPORT_MINUS_OTHER_SUPPORT_CELL_BOUNDS",
            context_ownership_confirmed=False,
        )


def _mixed_support_slice(source, unique_supports, xyz, support_bounds, cell_m):
    """Recover only points that cannot belong to any other original support.

    B1 has already merged these support IDs into one unresolved source. Its
    cell bounds are conservative necessary membership conditions. Removing
    every other support's bounds proves a subset of the unique support union;
    overlapping and missing-bound cases remain unattributed.
    """
    ids = set(source.public["source_support_ids"])
    allowed = ids.intersection(unique_supports)
    if not allowed or xyz is None or support_bounds is None or cell_m is None:
        return None
    if any(ident not in support_bounds for ident in ids):
        return None
    indexes = source.point_indexes
    # Membership follows B1's float32 cell assignment. Geometry retains the
    # original unmodified Raw values indexed by the resulting immutable slice.
    xy = np.asarray(xyz[indexes, :2], dtype=np.float32)

    def inside(ident):
        bounds = np.asarray(support_bounds[ident], dtype=float)
        return np.all(
            (xy >= bounds[:2] - cell_m / 2) & (xy <= bounds[2:] + cell_m / 2), axis=1
        )

    keep = np.zeros(len(indexes), dtype=bool)
    for ident in sorted(allowed):
        keep |= inside(ident)
    for ident in sorted(ids - allowed):
        keep &= ~inside(ident)
    values = np.frombuffer(
        np.asarray(indexes[keep], dtype="<i8").tobytes(), dtype="<i8"
    )
    if not len(values):
        return None
    return BoundarySourceSlice(
        source,
        values,
        hashlib.sha256(memoryview(values).cast("B")).hexdigest(),
        tuple(sorted(allowed)),
    )


@dataclass(frozen=True)
class BoundaryMeasurementContext:
    vessel_id: str
    sources: tuple
    point_indexes: np.ndarray
    point_source_codes: np.ndarray
    point_index_sha256: str
    unattributed_sources: tuple = ()

    @classmethod
    def build(cls, vessel, vessels, xyz=None, support_bounds=None, cell_m=None):
        """Retain unresolved sibling measurements without changing ownership.

        A spatial support can contain several physical bodies. Siblings are
        contextual only when their support has exactly one resolved core; a
        second resolved core or an unrelated support cannot be borrowed.
        """
        vessels = tuple(vessels)
        supports = set(vessel.public["source_support_ids"])
        unattributed = tuple(
            sorted(
                (
                    other
                    for other in vessels
                    if other.vessel_id != vessel.vessel_id
                    and other.public["classification_status"] == "UNRESOLVED"
                    and supports.intersection(other.public["source_support_ids"])
                ),
                key=lambda other: other.vessel_id,
            )
        )
        unique_supports = set(supports)
        if vessel.public["classification_status"] != "VESSEL_HYPOTHESIS":
            unique_supports.clear()
        for other in vessels:
            if (
                other.vessel_id != vessel.vessel_id
                and other.public["classification_status"] == "VESSEL_HYPOTHESIS"
            ):
                unique_supports.difference_update(other.public["source_support_ids"])
        siblings = tuple(
            sorted(
                (
                    other
                    for other in vessels
                    if other.vessel_id != vessel.vessel_id
                    and other.public["classification_status"] == "UNRESOLVED"
                    and other.public["source_support_ids"]
                    and set(other.public["source_support_ids"]).issubset(
                        unique_supports
                    )
                ),
                key=lambda other: other.vessel_id,
            )
        )
        slices = []
        sibling_ids = {s.vessel_id for s in siblings}
        for other in unattributed:
            if other.vessel_id in sibling_ids:
                continue
            part = _mixed_support_slice(
                other, unique_supports, xyz, support_bounds, cell_m
            )
            if part is not None:
                slices.append(part)
        siblings = tuple(sorted(siblings + tuple(slices), key=lambda s: s.vessel_id))
        sources = (vessel,) + siblings
        indexes = np.concatenate([source.point_indexes for source in sources])
        order = np.argsort(indexes, kind="stable")
        indexes = indexes[order]
        if len(np.unique(indexes)) != len(indexes):
            raise ValueError("BOUNDARY_CONTEXT_OWNERSHIP_OVERLAP")
        codes = np.concatenate(
            [
                np.full(len(source.point_indexes), code, dtype="<i4")
                for code, source in enumerate(sources)
            ]
        )[order]
        indexes = np.frombuffer(np.asarray(indexes, dtype="<i8").tobytes(), dtype="<i8")
        codes = np.frombuffer(codes.tobytes(), dtype="<i4")
        return cls(
            vessel.vessel_id,
            sources,
            indexes,
            codes,
            hashlib.sha256(memoryview(indexes).cast("B")).hexdigest(),
            unattributed,
        )

    @classmethod
    def build_all(cls, vessels, xyz=None, support_bounds=None, cell_m=None):
        # Unrelated spatial supports need no pairwise vessel comparison.
        by_support = {}
        for vessel in vessels:
            for support in vessel.public["source_support_ids"]:
                by_support.setdefault(support, {})[vessel.vessel_id] = vessel
        contexts = []
        for vessel in vessels:
            neighbors = {}
            for support in vessel.public["source_support_ids"]:
                neighbors.update(by_support[support])
            contexts.append(
                cls.build(vessel, neighbors.values(), xyz, support_bounds, cell_m)
            )
        return tuple(contexts)

    def identity(self):
        return dict(
            vessel_id=self.vessel_id,
            point_count=len(self.point_indexes),
            point_index_sha256=self.point_index_sha256,
            sources=[source.identity() for source in self.sources],
            unattributed_sources=[
                source.identity() for source in self.unattributed_sources
            ],
            source_code_sha256=hashlib.sha256(
                memoryview(self.point_source_codes).cast("B")
            ).hexdigest(),
            context_ownership_confirmed=False,
            method=(
                "OWNED_CORE_AND_UNRESOLVED_SIBLINGS_OF_UNIQUE_SUPPORT_CORE"
                + (
                    "_WITH_CONSERVATIVE_MIXED_SUPPORT_SLICES"
                    if any(
                        isinstance(source, BoundarySourceSlice)
                        for source in self.sources
                    )
                    else ""
                )
            ),
        )


@dataclass(frozen=True)
class SceneEvidenceContext:
    input_semantic_sha256: str
    raw_identity_json: str
    config_sha256: str
    software_sha: str
    scene: object
    vessels: tuple
    boundary_contexts: tuple

    @classmethod
    def build(cls, frame, config, research, config_sha, software_sha):
        scene, private = infer_scene_vessels(frame.xyz, config, research)
        vessels = []
        for row in scene["vessel_hypotheses"]:
            ownership_bytes = np.asarray(
                private["vessel_point_indexes"][row["vessel_hypothesis_id"]],
                dtype="<i8",
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
            BoundaryMeasurementContext.build_all(
                vessels,
                frame.xyz,
                {
                    s["support_id"]: s["bbox_xy"]
                    for s in scene.get("scene_support_candidates", [])
                },
                private["grid"].cell_m if private.get("grid") is not None else None,
            ),
        )

    def boundary_context(self, vessel_id):
        return next(
            context
            for context in self.boundary_contexts
            if context.vessel_id == vessel_id
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
            boundary_measurement_contexts=[
                context.identity() for context in self.boundary_contexts
            ],
        )
