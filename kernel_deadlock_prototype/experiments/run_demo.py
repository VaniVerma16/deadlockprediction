from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deadlock_prototype.events import EventLogger, TrackedLock


def run_safe(logger: EventLogger) -> None:
    """Run a workload with normal lock acquisition and release."""
    first = TrackedLock("lock:A", logger)
    second = TrackedLock("lock:B", logger)

    def worker() -> None:
        for _ in range(3):
            with first:
                time.sleep(0.01)

            with second:
                time.sleep(0.005)

    threads = [
        threading.Thread(
            target=worker,
            name=f"safe-worker-{i}",
        )
        for i in range(2)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def run_contention(logger: EventLogger) -> None:
    """Run a workload where multiple threads contend for one lock."""
    lock = TrackedLock("lock:contended", logger)
    barrier = threading.Barrier(3)

    def worker() -> None:
        barrier.wait()

        with lock:
            time.sleep(0.12)

    threads = [
        threading.Thread(
            target=worker,
            name=f"contention-worker-{i}",
        )
        for i in range(3)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def run_deadlock(logger: EventLogger) -> None:
    """
    Run a controlled two-lock deadlock scenario.

    Each thread acquires one lock and then waits for the other.
    A timeout is used so the demonstration can eventually recover.
    """
    first = TrackedLock("lock:A", logger)
    second = TrackedLock("lock:B", logger)

    barrier = threading.Barrier(2)

    def worker(
        own: TrackedLock,
        wait_for: TrackedLock,
    ) -> None:
        acquired = own.acquire()

        if not acquired:
            return

        try:
            barrier.wait()

            wait_for.acquire(timeout=0.25)

        finally:
            own.release()

    threads = [
        threading.Thread(
            target=worker,
            args=(first, second),
            name="deadlock-worker-0",
            daemon=True,
        ),
        threading.Thread(
            target=worker,
            args=(second, first),
            name="deadlock-worker-1",
            daemon=True,
        ),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join(timeout=0.4)


WORKLOADS = {
    "safe": run_safe,
    "contention": run_contention,
    "deadlock": run_deadlock,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run workload and collect raw runtime events"
    )

    parser.add_argument(
        "--mode",
        choices=sorted(WORKLOADS),
        default="safe",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "snapshots",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = args.output_dir / f"{args.mode}-events.jsonl"

    logger = EventLogger()

    WORKLOADS[args.mode](logger)

    if not logger.events:
        raise RuntimeError(
            "the workload produced no runtime events"
        )

    logger.write_jsonl(raw_path)

    print(
        f"Collected {len(logger.events)} raw events "
        f"for '{args.mode}' workload."
    )

    print(f"RAW_EVENTS={raw_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())