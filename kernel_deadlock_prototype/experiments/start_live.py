#!/usr/bin/env python3
"""Launch the native workload only after the live backend is listening."""
import argparse
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def capture(process, messages, label):
    for line in process.stdout:
        print(f"[{label}] {line}", end="", flush=True)
        messages.put(line.strip())


def wait_for(process, messages, predicate, timeout, label):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited with code {process.returncode}; see output above.")
        try:
            line = messages.get(timeout=0.1)
        except queue.Empty:
            continue
        if predicate(line):
            return
    raise RuntimeError(f"Timed out waiting for {label}; workload was not released.")


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["safe", "contention", "deadlock"], default="deadlock")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args()
    if sys.platform != "linux" or os.geteuid() != 0:
        parser.error('Run on Linux with: sudo env -u PYTHONPATH "$(command -v python3)" experiments/start_live.py')
    binary = ROOT / "build" / "pthread_workload"
    binary.parent.mkdir(exist_ok=True)
    subprocess.run(["cc", "-O0", "-g", "-pthread",
                    str(ROOT / "experiments" / "pthread_workload.c"), "-o", str(binary)], check=True)
    workload = backend = None
    try:
        # Drop the workload to the invoking user's privileges when launched via sudo.
        workload_options = {}
        if os.environ.get("SUDO_UID") and os.environ.get("SUDO_GID"):
            workload_options = dict(user=int(os.environ["SUDO_UID"]),
                                    group=int(os.environ["SUDO_GID"]), extra_groups=[])
        workload = subprocess.Popen([str(binary), args.mode, "--wait-start"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, **workload_options)
        workload_messages = queue.Queue()
        threading.Thread(target=capture, args=(workload, workload_messages, "workload"), daemon=True).start()
        wait_for(workload, workload_messages, lambda s: s == "WORKLOAD_READY", 10, "workload")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        boot = ("import sys,runpy; "
                "sys.path.append('/usr/lib/python3/dist-packages'); "
                "import websockets; "
                "print('WebSockets:', websockets.__version__, websockets.__file__, flush=True); "
                "runpy.run_path('server/live_server.py', run_name='__main__')")
        backend = subprocess.Popen([sys.executable, "-u", "-c", boot,
                                    "--pid", str(workload.pid), "--host", "127.0.0.1",
                                    "--port", str(args.port)], cwd=ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        backend_messages = queue.Queue()
        threading.Thread(target=capture, args=(backend, backend_messages, "backend"), daemon=True).start()
        wait_for(backend, backend_messages, lambda s: s.startswith("[ws] listening on "),
                 args.startup_timeout, "backend")
        if workload.poll() is not None or backend.poll() is not None:
            raise RuntimeError("A process exited before workload release.")
        print(f"Monitoring ready for PID {workload.pid}. Starting {args.mode} workload.", flush=True)
        workload.stdin.write("\n")
        workload.stdin.flush()
        workload.stdin.close()
        print("Keep this terminal open. Ctrl+C stops both processes.", flush=True)
        reported_exit = False
        while backend.poll() is None:
            if workload.poll() is not None and not reported_exit:
                print("Workload finished. Backend remains open for inspection; Ctrl+C to stop.", flush=True)
                reported_exit = True
            time.sleep(0.2)
        raise RuntimeError(f"Backend exited with code {backend.returncode}.")
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"Launch failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stop(workload)
        stop(backend)


if __name__ == "__main__":
    raise SystemExit(main())
