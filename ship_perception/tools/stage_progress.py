"""Console-only stage timing, independent of detection and acceptance logic."""

from contextlib import contextmanager, nullcontext
import sys
import threading
import time
from typing import Iterator, Optional, TextIO


class StageProgress:
    """Report elapsed monotonic time and heartbeats without estimated percentages."""

    def __init__(self, stream: Optional[TextIO] = None, interval_sec: float = 15.0):
        self.stream = sys.stdout if stream is None else stream
        self.interval_sec = interval_sec
        self.started = time.perf_counter()
        self.sequence = 0
        self.lock = threading.Lock()

    def _write(self, message: str) -> None:
        with self.lock:
            print(message, file=self.stream, flush=True)

    def _heartbeat(self, stop: threading.Event, label: str, started: float) -> None:
        while not stop.wait(self.interval_sec):
            self._write(
                "{} RUNNING stage_elapsed={:.2f}s total_elapsed={:.2f}s".format(
                    label,
                    time.perf_counter() - started,
                    time.perf_counter() - self.started,
                )
            )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.sequence += 1
        label = "[{:02d}] {}".format(self.sequence, name)
        started = time.perf_counter()
        stop = threading.Event()
        thread = None
        self._write(
            "{} START total_elapsed={:.2f}s".format(label, started - self.started)
        )
        if self.interval_sec > 0:
            thread = threading.Thread(
                target=self._heartbeat, args=(stop, label, started), daemon=True
            )
            thread.start()
        state = "DONE"
        try:
            yield
        except BaseException:
            state = "FAILED"
            raise
        finally:
            stop.set()
            if thread is not None:
                thread.join()
            ended = time.perf_counter()
            self._write(
                "{} {} stage_elapsed={:.2f}s total_elapsed={:.2f}s".format(
                    label, state, ended - started, ended - self.started
                )
            )

    def finish(self, success: bool) -> None:
        self._write(
            "TOTAL status={} elapsed={:.2f}s".format(
                "COMPLETED" if success else "FAILED", time.perf_counter() - self.started
            )
        )


def stage(progress: Optional[StageProgress], name: str):
    """Keep library calls silent unless the caller explicitly requests progress."""
    return nullcontext() if progress is None else progress.stage(name)
