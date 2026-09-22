#!/usr/bin/env python3
"""Small Linux multithreaded application for testing the live eBPF monitor."""
from __future__ import annotations
import argparse
import os
import threading
import time


def safe_worker(lock: threading.Lock, duration: float) -> None:
    end = time.monotonic() + duration
    while time.monotonic() < end:
        with lock:
            time.sleep(0.005)
        time.sleep(0.005)


def contention_worker(lock: threading.Lock, duration: float) -> None:
    end = time.monotonic() + duration
    while time.monotonic() < end:
        with lock:
            time.sleep(0.030)
        time.sleep(0.005)


def run_safe(duration: float) -> None:
    locks = [threading.Lock(), threading.Lock()]
    threads = [threading.Thread(target=safe_worker, args=(lock, duration)) for lock in locks]
    for t in threads: t.start()
    for t in threads: t.join()


def run_contention(duration: float) -> None:
    lock = threading.Lock()
    threads = [threading.Thread(target=contention_worker, args=(lock, duration)) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()


def run_deadlock(duration: float) -> None:
    lock_a = threading.Lock()
    lock_b = threading.Lock()
    ready_a = threading.Event()
    ready_b = threading.Event()

    def worker_a() -> None:
        with lock_a:
            ready_a.set()
            ready_b.wait()
            lock_b.acquire()

    def worker_b() -> None:
        with lock_b:
            ready_b.set()
            ready_a.wait()
            lock_a.acquire()

    threading.Thread(target=worker_a).start()
    threading.Thread(target=worker_b).start()
    time.sleep(duration)
    print("Deadlock workload is active; press Ctrl+C to stop.")
    while True:
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['safe', 'contention', 'deadlock'], default='safe')
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()
    print(f'PID={os.getpid()}')
    print(f'Mode={args.mode}')
    print(f'Duration={args.duration}s')
    if args.mode == 'safe': run_safe(args.duration)
    elif args.mode == 'contention': run_contention(args.duration)
    else: run_deadlock(args.duration)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
