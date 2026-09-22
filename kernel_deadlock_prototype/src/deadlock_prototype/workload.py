from __future__ import annotations

import threading
import time

from .events import EventLogger, TrackedLock


def run_safe_workload(logger: EventLogger) -> None:
    first = TrackedLock("lock:A", logger)
    second = TrackedLock("lock:B", logger)

    def worker() -> None:
        for _ in range(3):
            with first:
                time.sleep(0.01)
            with second:
                time.sleep(0.005)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run_contention_workload(logger: EventLogger) -> None:
    lock = TrackedLock("lock:contended", logger)
    barrier = threading.Barrier(3)

    def worker() -> None:
        barrier.wait()
        with lock:
            time.sleep(0.12)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run_deadlock_workload(logger: EventLogger) -> None:
    first = TrackedLock("lock:A", logger)
    second = TrackedLock("lock:B", logger)
    barrier = threading.Barrier(2)

    def worker(own: TrackedLock, wait_for: TrackedLock) -> None:
        own.acquire()
        try:
            barrier.wait()
            wait_for.acquire(timeout=0.25)
        finally:
            own.release()

    threads = [
        threading.Thread(target=worker, args=(first, second), daemon=True),
        threading.Thread(target=worker, args=(second, first), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=0.4)


WORKLOADS = {
    "safe": run_safe_workload,
    "contention": run_contention_workload,
    "deadlock": run_deadlock_workload,
}