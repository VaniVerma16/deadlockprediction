# Linux VM Notes

The Python-only demo does not require root, eBPF, `perf`, or `strace`. Run it first to verify the event and graph contracts.

For later instrumentation, install optional tools in Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip gcc make strace linux-tools-common linux-tools-$(uname -r)
```

Use `perf stat -e context-switches,cpu-migrations` around a workload to collect coarse process-level counters. Keep those counters separate from per-thread graph features until they are sampled per time window.

The next collector stage should translate eBPF or tracepoint records into the same event fields used by `RuntimeEvent`: `ts_ns`, `event`, `tid`, `lock_id`, `owner_tid`, `wait_ns`, `wakeups`, `context_switches`, and `cpu_migrations`.
