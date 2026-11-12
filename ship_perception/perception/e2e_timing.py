"""Read-only timing at geometry readiness, separate from rendering/publication."""

import copy
import time

from .r3h_geometry import dimensions


class ReviewTiming:
    def __init__(
        self, worker_start, read_start, read_end, mode, input_sha256, input_bytes
    ):
        self.worker_start = worker_start
        self.read_start = read_start
        self.read_end = read_end
        self.mode = mode
        self.input_sha256 = input_sha256
        self.input_bytes = input_bytes
        self.ready_at = None
        self.export_at = None
        self.objects = []
        self.extraction_s = None

    def geometry_ready(self, payload):
        started = time.perf_counter()
        rows = []
        for cell in payload["geometry_only"].get("selected_opening_cells", []):
            strict = cell.get("geometry")
            if strict is None:
                rows.append(
                    dict(
                        cell_id=cell["cell_id"],
                        state="UNRESOLVED",
                        corners_xyz=None,
                        length_m=None,
                        width_m=None,
                    )
                )
                continue
            review = cell.get("review_geometry", {})
            current = review if review.get("adopted_for_review") else strict
            rows.append(
                dict(
                    cell_id=cell["cell_id"],
                    vessel_id=cell["vessel_id"],
                    state=review.get("state", "RETAINED_STRICT_BASELINE"),
                    strict_rectangle_state=cell["strict_rectangle_state"],
                    corners_xyz=copy.deepcopy(current["corners_xyz"]),
                    **dimensions(current["boundary_xy"]),
                    steel_certified=False,
                    control_eligible=False,
                )
            )
        self.ready_at = time.perf_counter()
        self.extraction_s = self.ready_at - started
        self.objects = rows

    def published(self):
        self.export_at = time.perf_counter()

    def report(self):
        return dict(
            schema="ship_perception.e2e_timing.1",
            clock="time.perf_counter",
            mode=self.mode,
            is_full_phase=self.mode == "full",
            input_sha256=self.input_sha256,
            input_file_bytes=self.input_bytes,
            read_pcd_s=self.read_end - self.read_start,
            read_start_to_geometry_ready_s=(
                None if self.ready_at is None else self.ready_at - self.read_start
            ),
            worker_start_to_geometry_ready_s=(
                None if self.ready_at is None else self.ready_at - self.worker_start
            ),
            readiness_extraction_s=self.extraction_s,
            read_start_to_publication_s=(
                None if self.export_at is None else self.export_at - self.read_start
            ),
            geometry_ready_to_publication_s=(
                None
                if self.ready_at is None or self.export_at is None
                else self.export_at - self.ready_at
            ),
            geometry_ready=self.ready_at is not None,
            publication_complete=self.export_at is not None,
            geometry_readiness_semantic="ALL_REVIEW_XYZ_AND_DIMENSIONS_AVAILABLE_BEFORE_RENDER;NOT_STEEL_CERTIFICATION",
            objects=self.objects,
        )
