"""Process watchdog and finite measurement progress; no geometry decisions."""

import copy
import faulthandler
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import time
import uuid


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _watch(parent_pid, stop, folder, soft, hard):
    parent = multiprocessing.parent_process()
    start = time.monotonic()
    slow = False
    handle = None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000 | 0x0001, False, parent_pid)
        if not handle:
            _atomic(
                Path(folder) / "watchdog_error.json", dict(status="WATCHDOG_ARM_FAILED")
            )
            return
    try:
        while not stop.wait(min(0.25, hard)):
            if not parent or not parent.is_alive():
                return
            elapsed = time.monotonic() - start
            if elapsed >= soft and not slow:
                _atomic(
                    Path(folder) / "slow_scene.json",
                    dict(SLOW_SCENE=True, elapsed_s=elapsed),
                )
                slow = True
            if elapsed >= hard:
                checkpoint = Path(folder) / "checkpoint.json"
                state = (
                    json.loads(checkpoint.read_text(encoding="utf-8"))
                    if checkpoint.exists()
                    else {}
                )
                _atomic(
                    Path(folder) / "RUN_TIMEOUT.json",
                    dict(
                        status="RUN_TIMEOUT",
                        elapsed_s=elapsed,
                        checkpoint=state,
                        watchdog_hit=True,
                        soft_runtime_limit_s=soft,
                        hard_process_watchdog_s=hard,
                        traceback=str(Path(folder) / "traceback.log"),
                        no_hatch_verdict=False,
                    ),
                )
                if os.name == "nt":
                    kernel.TerminateProcess(handle, 124)
                else:
                    os.kill(parent_pid, signal.SIGKILL)
                return
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)


class ReviewRunMonitor:
    def __init__(self, output_dir, identity=None, soft=120.0, hard=300.0):
        self.folder = Path(output_dir) / (".review_run_" + uuid.uuid4().hex)
        self.folder.mkdir(parents=True)
        self.soft, self.hard = soft, hard
        self.started = time.monotonic()
        self.state = dict(
            status="RUNNING",
            stage="START",
            identity=identity or {},
            tasks={},
            cache_state={},
        )
        self.process = None
        self.stop = None
        self.trace = None

    def __enter__(self):
        self.checkpoint()
        ctx = multiprocessing.get_context("spawn")
        self.stop = ctx.Event()
        self.trace = (self.folder / "traceback.log").open("w", encoding="utf-8")
        faulthandler.dump_traceback_later(max(0.01, self.hard - 0.1), file=self.trace)
        self.process = ctx.Process(
            target=_watch,
            args=(os.getpid(), self.stop, str(self.folder), self.soft, self.hard),
            daemon=True,
        )
        self.process.start()
        return self

    def checkpoint(self, stage=None, tasks=None, cache_state=None, review_objects=None):
        if stage is not None:
            self.state["stage"] = stage
        if tasks is not None:
            self.state["tasks"] = copy.deepcopy(tasks)
        if cache_state is not None:
            self.state["cache_state"] = copy.deepcopy(cache_state)
        if review_objects is not None:
            self.state["review_objects"] = copy.deepcopy(review_objects)
        self.state["elapsed_s"] = time.monotonic() - self.started
        _atomic(self.folder / "checkpoint.json", self.state)

    def __exit__(self, kind, value, traceback):
        self.state["status"] = "FAILED" if kind else "COMPLETED"
        self.checkpoint()
        faulthandler.cancel_dump_traceback_later()
        if self.stop:
            self.stop.set()
        if self.process:
            self.process.join(timeout=2)
        if self.trace:
            self.trace.close()


class MeasurementTasks:
    def __init__(self, input_identity, monitor=None):
        self.input_identity = input_identity
        self.monitor = monitor
        self.rows = {}

    def begin(self, region_id, side, signature):
        task_id = hashlib.sha256(
            (self.input_identity + "\0" + region_id + "\0" + side).encode()
        ).hexdigest()
        old = self.rows.get(task_id)
        if old and old["progress_signature"] == signature:
            old["status"] = "NO_PROGRESS"
            return None
        self.rows[task_id] = dict(
            task_id=task_id,
            input_identity=self.input_identity,
            region_id=region_id,
            side=side,
            attempt_count=1 if old is None else old["attempt_count"] + 1,
            progress_signature=signature,
            status="RUNNING",
        )
        self.flush()
        return task_id

    def finish(self, task_id, status, signature, cache_state=None):
        self.rows[task_id].update(status=status, progress_signature=signature)
        self.flush(cache_state)

    def flush(self, cache_state=None):
        if self.monitor:
            self.monitor.checkpoint(
                stage="BOUNDARY_MEASUREMENT", tasks=self.rows, cache_state=cache_state
            )
