#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from deadlock_prototype.events import RuntimeEvent


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>

struct event_t {
    u64 ts_ns;
    u32 pid;
    u32 tid;
    u32 type;
    u64 lock_ptr;
    s64 ret;
    u32 cpu;
};

BPF_PERF_OUTPUT(events);

BPF_HASH(lock_pending, u32, u64);
BPF_HASH(trylock_pending, u32, u64);

enum {
    EVT_LOCK_WAIT_START = 1,
    EVT_LOCK_ACQUIRED = 2,
    EVT_LOCK_RELEASED = 3,
    EVT_LOCK_TRY_FAILED = 4,
    EVT_FUTEX_WAIT = 5,
    EVT_FUTEX_WAKE = 6,
    EVT_SCHED_SWITCH = 7
};

static int emit_event(struct pt_regs *ctx, u32 type, u64 lock_ptr, s64 ret) {
    struct event_t event = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();

    event.pid = pid_tgid >> 32;
    event.tid = (u32)pid_tgid;
    event.ts_ns = bpf_ktime_get_ns();
    event.type = type;
    event.lock_ptr = lock_ptr;
    event.ret = ret;
    event.cpu = bpf_get_smp_processor_id();

    events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}


int lock_enter(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 lock_ptr = PT_REGS_PARM1(ctx);

    lock_pending.update(&tid, &lock_ptr);

    return emit_event(
        ctx,
        EVT_LOCK_WAIT_START,
        lock_ptr,
        0
    );
}


int lock_return(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 *lock_ptr = lock_pending.lookup(&tid);

    if (!lock_ptr)
        return 0;

    s64 ret = PT_REGS_RC(ctx);

    if (ret == 0) {
        emit_event(
            ctx,
            EVT_LOCK_ACQUIRED,
            *lock_ptr,
            ret
        );
    } else {
        emit_event(
            ctx,
            EVT_LOCK_TRY_FAILED,
            *lock_ptr,
            ret
        );
    }

    lock_pending.delete(&tid);
    return 0;
}


int trylock_enter(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 lock_ptr = PT_REGS_PARM1(ctx);

    trylock_pending.update(&tid, &lock_ptr);

    return emit_event(
        ctx,
        EVT_LOCK_WAIT_START,
        lock_ptr,
        0
    );
}


int trylock_return(struct pt_regs *ctx) {
    u32 tid = (u32)bpf_get_current_pid_tgid();
    u64 *lock_ptr = trylock_pending.lookup(&tid);

    if (!lock_ptr)
        return 0;

    s64 ret = PT_REGS_RC(ctx);

    if (ret == 0) {
        emit_event(
            ctx,
            EVT_LOCK_ACQUIRED,
            *lock_ptr,
            ret
        );
    } else {
        emit_event(
            ctx,
            EVT_LOCK_TRY_FAILED,
            *lock_ptr,
            ret
        );
    }

    trylock_pending.delete(&tid);
    return 0;
}


int unlock_enter(struct pt_regs *ctx) {
    u64 lock_ptr = PT_REGS_PARM1(ctx);

    return emit_event(
        ctx,
        EVT_LOCK_RELEASED,
        lock_ptr,
        0
    );
}


/*
 * Futex commands:
 *
 * FUTEX_WAIT = 0
 * FUTEX_WAKE = 1
 *
 * The low 4 bits contain the command.
 */


"""


class EBPFCollector:

    def __init__(self, pid: int | None = None):

        try:
            from bcc import BPF
        except ImportError as exc:
            raise RuntimeError(
                "BCC is not installed. Run on Linux with BCC available."
            ) from exc

        self.BPF = BPF
        self.pid = pid

        self.bpf = BPF(text=BPF_PROGRAM)

        self.events: list[RuntimeEvent] = []

        self.libc = self._find_libc()

        self._attach_uprobe(
            "pthread_mutex_lock",
            "lock_enter"
        )

        self._attach_uretprobe(
            "pthread_mutex_lock",
            "lock_return"
        )

        self._attach_uprobe(
            "pthread_mutex_trylock",
            "trylock_enter"
        )

        self._attach_uretprobe(
            "pthread_mutex_trylock",
            "trylock_return"
        )

        self._attach_uprobe(
            "pthread_mutex_unlock",
            "unlock_enter"
        )

        self.bpf["events"].open_perf_buffer(
            self._handle_event
        )


    @staticmethod
    def _find_libc() -> str:
        libc = "/usr/lib/x86_64-linux-gnu/libc.so.6"

        if not Path(libc).exists():
            raise RuntimeError(
                f"Could not locate libc at {libc}"
            )

        return libc

    def _attach_uprobe(
        self,
        symbol: str,
        fn: str
    ) -> None:

        self.bpf.attach_uprobe(
            name=self.libc,
            sym=symbol,
            fn_name=fn,
            pid=self.pid if self.pid is not None else -1,
        )


    def _attach_uretprobe(
        self,
        symbol: str,
        fn: str
    ) -> None:

        self.bpf.attach_uretprobe(
            name=self.libc,
            sym=symbol,
            fn_name=fn,
            pid=self.pid if self.pid is not None else -1,
        )


    def _handle_event(
        self,
        cpu,
        data,
        size
    ) -> None:

        event = self.bpf["events"].event(data)

        if self.pid is not None:
            if int(event.pid) != self.pid:
                return

        event_names = {
            1: "lock_wait_start",
            2: "lock_acquired",
            3: "lock_released",
            4: "lock_try_failed",
            5: "futex_wait",
            6: "futex_wake",
            7: "sched_switch",
        }

        event_name = event_names.get(
            int(event.type),
            f"unknown_{event.type}"
        )

        lock_id = None

        if event.lock_ptr:
            lock_id = (
                f"lock:{event.pid}:"
                f"0x{event.lock_ptr:x}"
            )

        runtime_event = RuntimeEvent(
            ts_ns=int(event.ts_ns),
            event=event_name,
            tid=int(event.tid),
            lock_id=lock_id,
            owner_tid=None,
            wait_ns=None,
            wakeups=0.0,
            context_switches=0,
            cpu_migrations=0,
            cpu=int(event.cpu),
        )

        self.events.append(runtime_event)

        print(
            f"{event_name:18s} "
            f"pid={event.pid} "
            f"tid={event.tid} "
            f"lock={lock_id}"
        )


    def poll(
        self,
        timeout_ms: int = 100
    ) -> None:

        self.bpf.perf_buffer_poll(
            timeout=timeout_ms
        )


    def run(self) -> None:

        print(
            "eBPF collector attached; "
            f"PID={self.pid if self.pid is not None else 'all'}"
        )

        try:

            while True:
                self.poll()

        except KeyboardInterrupt:

            print(
                "\nStopping eBPF collector."
            )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pid",
        type=int,
        default=None
    )

    args = parser.parse_args()

    collector = EBPFCollector(
        pid=args.pid
    )

    collector.run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())