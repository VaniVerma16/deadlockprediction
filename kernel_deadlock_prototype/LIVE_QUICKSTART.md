# One-command live workload and monitor

Use Linux with BCC, kernel tracing support, a C compiler, the existing model checkpoint, and the project Python dependencies installed. Keep your existing Python environment activated.

From kernel_deadlock_prototype:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install --upgrade "websockets>=13,<16"
sudo env -u PYTHONPATH "$(command -v python3)" experiments/start_live.py --mode deadlock
```

Use --mode safe or --mode contention for the other workloads. Stop an earlier backend before launching another on the same port.

The launcher compiles the native workload, starts it behind an input gate, captures its PID automatically, and starts the backend with that PID. It releases the workload only after the backend reports that it is listening, which follows probe attachment and model initialization. A startup failure or timeout cancels the workload. Ctrl+C stops both processes. Safe and contention workloads finish naturally; the backend remains available until Ctrl+C.

The launcher requires root for the eBPF backend, but drops the workload to SUDO_UID/SUDO_GID when invoked through sudo. It preserves the selected Python executable and appends system Python packages for BCC after the environment packages, avoiding the older system websockets package taking precedence.

Start the frontend separately and leave it running across demo runs:

```bash
cd frontend
npm install
VITE_WS_URL=ws://127.0.0.1:8000/ws npm run dev
```

Open http://localhost:5173 on the same computer. The frontend reconnects automatically when the backend restarts.

Options: --port 8000, --startup-timeout 120 (seconds). If using a different port, change VITE_WS_URL to match.

Validation: native workload compilation and gated-start behavior tested; launcher readiness and Ctrl+C behavior tested with a stub backend. Full privileged eBPF/model integration must be verified on the target Linux computer.
