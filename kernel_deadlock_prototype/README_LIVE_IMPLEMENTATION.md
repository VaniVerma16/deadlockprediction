# Live Synchronization Digital Twin + V5

This adds the live implementation layer on top of the existing prototype.

## Architecture

Running Linux application
    -> eBPF Uprobes/Uretprobes
    -> Futex/Scheduler telemetry
    -> normalized RuntimeEvent
    -> Synchronization Digital Twin
    -> fixed 10-ms TSG snapshots
    -> existing V5 GraphSAGE + GRU inference
    -> Safe / Pre-Deadlock / Deadlocked + risk

## Files

- `src/deadlock_prototype/digital_twin.py`
  Continuous synchronization Digital Twin and 10-ms sampler.

- `experiments/ebpf_collector.py`
  Linux BCC/eBPF telemetry collector.

- `experiments/live_monitor.py`
  End-to-end collector -> Digital Twin -> V5 monitor.

- `requirements-live.txt`
  Python/runtime dependencies and Linux BCC installation notes.

## Important

The eBPF collector is Linux-only. eBPF cannot be tested directly on macOS.
The existing prototype workloads can still be used on macOS for pipeline
development and model integration.

## Running

From `kernel_deadlock_prototype` on Linux:

1. Make sure the existing V5 model checkpoint is accessible.
2. Start the multithreaded application you want to monitor.
3. Obtain its PID.
4. Run:

    sudo -E python3 experiments/live_monitor.py --pid <PID>

The monitor samples the Digital Twin every 10 ms and feeds each graph
snapshot into the existing V5 inference buffer.

## Instrumentation note

`pthread_mutex_*` functions are instrumented at the application/library
boundary. Futex and scheduler events provide kernel-assisted contention and
scheduling evidence.

The Digital Twin reconstructs synchronization state from the normalized
events. The TSG is the internal graph representation of that state.

## Current prototype limitation

The eBPF collector is intentionally a first integration implementation.
Production-grade deployment should additionally handle:
- static/custom synchronization libraries,
- symbol availability differences across libc versions,
- richer futex wait/wake correlation,
- PID/TID lifecycle handling,
- event-loss accounting,
- per-process library resolution,
- lower-overhead event buffering.

Do not claim production-level overhead or complete kernel coverage from this
prototype alone.
