"""Interrupted preflight must remain attributable even without a full result."""

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ship_perception.perception.precision_diagnostics import (
    CandidateTrace,
    PrecisionDiagnostics,
    side_completed,
    stage,
)
from ship_perception.perception.budget import PerformanceBudgetExceeded
from ship_perception.perception.runtime_patch import git_source_state


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class PrecisionAccounting(unittest.TestCase):
    def test_interrupted_shared_investigation_is_not_zero_work(self):
        clock = Clock()
        trace = CandidateTrace("C", "Q", "R", (0, 0, 40, 10), clock)
        with self.assertRaises(PerformanceBudgetExceeded):
            with trace.visit():
                with trace.measure("anchor_role"):
                    clock.advance(2)
                    side_completed("U0", "RAW_SIDE_INVESTIGATION", "STRONG_UNIQUE_EDGE")
                    with trace.measure("normal"):
                        clock.advance(18)
                        raise PerformanceBudgetExceeded(
                            "PERFORMANCE_BUDGET_EXCEEDED:FINAL_PRECISION"
                        )
        self.assertEqual(trace.row["elapsed_total"], 20)
        self.assertEqual(trace.row["anchor_role_elapsed"], 2)
        self.assertEqual(trace.row["normal_elapsed"], 18)
        self.assertEqual(trace.row["component_inclusive_elapsed"]["anchor_role"], 20)
        self.assertEqual(trace.row["timeout_stage"], "normal")
        self.assertEqual(trace.row["sides_completed_before_timeout"], ["U0"])
        self.assertFalse(trace.row["full_precision_started"])
        self.assertFalse(trace.row["full_precision_completed"])
        self.assertEqual(trace.row["status"], "DEFERRED_PERFORMANCE")

    def test_all_candidates_remain_scheduled_and_only_completed_full_calls_count(self):
        diagnostics = PrecisionDiagnostics()
        queries = [
            SimpleNamespace(
                query_id="Q" + str(i),
                vessel_id="V",
                bounds=(i * 30, 0, i * 30 + 20, 10),
            )
            for i in range(3)
        ]
        traces = [diagnostics.schedule(q, {}) for q in queries]
        with traces[0].visit():
            traces[0].row["full_precision_started"] = True
            traces[0].row["full_precision_completed"] = True
        with traces[1].visit():
            pass  # A completed cheap probe is not a full precision completion.
        self.assertIs(diagnostics.schedule(queries[0], {}), traces[0])
        report = diagnostics.report()
        self.assertEqual(
            [
                report[k]
                for k in (
                    "precision_scheduled_count",
                    "precision_started_count",
                    "precision_completed_count",
                    "precision_timeout_count",
                    "precision_deferred_count",
                )
            ],
            [3, 2, 1, 0, 2],
        )
        self.assertEqual(len({t.row["physical_group_id"] for t in traces}), 3)

    def test_instrumentation_preserves_return_value_and_exception_without_active_trace(
        self,
    ):
        @stage("raw_profile")
        def primitive(value):
            if value is None:
                raise ValueError("NO_OBSERVATION")
            return value

        observation = {"rho": 1.234, "raw_ids": [9, 4, 6]}
        self.assertIs(primitive(observation), observation)
        with self.assertRaisesRegex(ValueError, "NO_OBSERVATION"):
            primitive(None)

    def test_saved_patch_includes_new_untracked_source_and_ignores_stale_stat_cache(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / "new.py").write_text("observed = True\n", encoding="utf-8")

            def command(args, **kwargs):
                if args[1] == "diff":
                    return b""
                if args[1] == "ls-files":
                    return b"new.py\0"
                if args[1] == "rev-parse":
                    return b"a" * 40 + b"\n"
                self.fail(str(args))

            with patch(
                "ship_perception.perception.runtime_patch.subprocess.check_output",
                side_effect=command,
            ):
                identity, diff = git_source_state(root)
            self.assertTrue(identity["git_worktree_dirty"])
            self.assertIn("+++ b/new.py", diff)
            self.assertIn("+observed = True", diff)
            self.assertEqual(
                identity["git_diff_sha256"], hashlib.sha256(diff.encode()).hexdigest()
            )
            with patch(
                "ship_perception.perception.runtime_patch.subprocess.check_output",
                side_effect=lambda args, **kwargs: (
                    b"a" * 40 + b"\n" if args[1] == "rev-parse" else b""
                ),
            ):
                identity, diff = git_source_state(root)
            self.assertFalse(identity["git_worktree_dirty"])
            self.assertEqual(diff, "")
