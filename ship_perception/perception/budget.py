"""Execution budgets are engineering limits, not geometric acceptance gates."""

from contextlib import contextmanager
import time


class PerformanceBudgetExceeded(RuntimeError):
    pass


class PerformanceBudget:
    LIMITS = {
        "SCENE_PREPROCESS": 3.0,
        "BERTH_SIDE_VESSEL_SELECTION": 2.0,
        "LOCAL_RECALL": 10.0,
        "UNIFIED_PHYSICAL_VALIDATION": 5.0,
        "FINAL_PRECISION": 20.0,
        "RENDER": 5.0,
    }

    def __init__(self, total_s=60.0):
        self.started = time.perf_counter()
        self.total_s = total_s
        self.events = []
        self.current = None
        self.deadline = None

    def check(self):
        now = time.perf_counter()
        if now - self.started > self.total_s or (
            self.deadline is not None and now > self.deadline
        ):
            raise PerformanceBudgetExceeded(
                "PERFORMANCE_BUDGET_EXCEEDED:" + str(self.current)
            )

    @contextmanager
    def stage(self, name, hard=False):
        previous = (self.current, self.deadline)
        self.current = name
        begin = time.perf_counter()
        limit = self.LIMITS.get(name, self.total_s)
        self.deadline = begin + limit if hard else None
        try:
            self.check()
            yield
        finally:
            elapsed = time.perf_counter() - begin
            self.events.append(
                dict(
                    stage=name,
                    elapsed_s=elapsed,
                    budget_s=limit,
                    status=(
                        "PERFORMANCE_BUDGET_EXCEEDED"
                        if elapsed > limit
                        else "WITHIN_BUDGET"
                    ),
                )
            )
            self.current, self.deadline = previous

    def report(self):
        elapsed = time.perf_counter() - self.started
        return dict(
            total_elapsed_s=elapsed,
            total_budget_s=self.total_s,
            total_budget_exceeded=elapsed > self.total_s,
            stages=list(self.events),
        )
