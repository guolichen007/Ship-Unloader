"""Account for cheap probes, shared investigations and interrupted refinement."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import time
from .budget import PerformanceBudgetExceeded

_active = ContextVar("ship_precision_candidate", default=None)
COMPONENTS = (
    "anchor_role",
    "raw_profile",
    "raw_edge_search",
    "height",
    "normal",
    "raster",
    "line_fit",
    "corner",
)


def stage(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            trace = _active.get()
            if trace is None:
                return function(*args, **kwargs)
            with trace.measure(name):
                return function(*args, **kwargs)

        return measured

    return decorate


def side_completed(name, phase, status):
    trace = _active.get()
    if trace is not None:
        trace.row["side_progress"].setdefault(name, {})[phase] = status


class CandidateTrace:
    def __init__(
        self, candidate_id, query_id, group_id, bounds, clock=time.perf_counter
    ):
        self.clock = clock
        self.row = dict(
            candidate_id=candidate_id,
            query_id=query_id,
            physical_group_id=group_id,
            bounds_axial=list(bounds),
            status="SCHEDULED",
            elapsed_total=0.0,
            visits=0,
            full_precision_started=False,
            full_precision_completed=False,
            timeout_stage=None,
            side_progress={},
            sides_completed_before_timeout=[],
            component_inclusive_elapsed={},
            component_call_count={},
        )
        self.row.update({name + "_elapsed": 0.0 for name in COMPONENTS})
        self.stack = []

    @contextmanager
    def measure(self, name):
        started = self.clock()
        frame = [name, started, 0.0]
        self.stack.append(frame)
        try:
            yield
        except Exception as error:
            if isinstance(error, PerformanceBudgetExceeded):
                self.row["timeout_stage"] = self.row["timeout_stage"] or name
            raise
        finally:
            elapsed = self.clock() - started
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += elapsed
            self.row[name + "_elapsed"] += max(0.0, elapsed - frame[2])
            inclusive = self.row["component_inclusive_elapsed"]
            inclusive[name] = inclusive.get(name, 0.0) + elapsed
            counts = self.row["component_call_count"]
            counts[name] = counts.get(name, 0) + 1

    @contextmanager
    def visit(self):
        started = self.clock()
        token = _active.set(self)
        self.row["visits"] += 1
        self.row["status"] = "STARTED"
        try:
            yield self
            if self.row["status"] == "STARTED":
                self.row["status"] = (
                    "COMPLETED"
                    if self.row["full_precision_completed"]
                    else "PROBED_NO_FULL_REFINEMENT"
                )
        except Exception as error:
            self.row["status"] = "INTERRUPTED"
            if isinstance(error, PerformanceBudgetExceeded):
                self.row["timeout_stage"] = self.row["timeout_stage"] or (
                    self.stack[-1][0] if self.stack else "BETWEEN_COMPONENTS"
                )
                self.row["sides_completed_before_timeout"] = sorted(
                    self.row["side_progress"]
                )
                self.row["status"] = "DEFERRED_PERFORMANCE"
            self.row["interruption_type"] = type(error).__name__
            raise
        finally:
            self.row["elapsed_total"] += self.clock() - started
            self.row["unattributed_elapsed"] = max(
                0.0,
                self.row["elapsed_total"]
                - sum(self.row[name + "_elapsed"] for name in COMPONENTS),
            )
            _active.reset(token)


class PrecisionDiagnostics:
    def __init__(self):
        self.traces = {}

    def schedule(self, query, row):
        bounds = list(query.bounds)
        candidate = (
            query.query_id
            + ":"
            + hashlib.sha256(json.dumps(bounds).encode()).hexdigest()[:12]
        )
        group = (
            row.get("physical_group_id")
            or "R"
            + hashlib.sha256(
                json.dumps(
                    [query.vessel_id, row.get("scheduler_region_bounds", bounds)]
                ).encode()
            ).hexdigest()[:16]
        )
        if candidate not in self.traces:
            self.traces[candidate] = CandidateTrace(
                candidate, query.query_id, group, bounds
            )
        return self.traces[candidate]

    def report(self):
        rows = [trace.row for trace in self.traces.values()]
        return dict(
            precision_scheduled_count=len(rows),
            precision_started_count=sum(r["visits"] > 0 for r in rows),
            precision_completed_count=sum(r["full_precision_completed"] for r in rows),
            precision_timeout_count=sum(
                r["status"] == "DEFERRED_PERFORMANCE" for r in rows
            ),
            precision_deferred_count=sum(
                not r["full_precision_completed"] for r in rows
            ),
            max_single_precision_time=max(
                (r["elapsed_total"] for r in rows), default=0.0
            ),
            counting_semantics="Unique candidate bounds; started includes RawProbe/shared role investigation; completed means full refiner returned, irrespective of geometry success; deferred includes scheduled but no completed full refinement",
            candidates=rows,
        )
