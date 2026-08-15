#ifndef DEADLOCK_EVENTS_H
#define DEADLOCK_EVENTS_H

#ifndef __VMLINUX_H__
#include <linux/types.h>
#endif

enum event_type {
    EVENT_LOCK_ATTEMPT = 1,
    EVENT_LOCK_ACQUIRED,
    EVENT_TRYLOCK_ATTEMPT,
    EVENT_TRYLOCK_ACQUIRED,
    EVENT_LOCK_RELEASED,
    EVENT_FUTEX_WAIT,
    EVENT_FUTEX_RETURN,
    EVENT_SCHED_SWITCH,
    EVENT_THREAD_WAKEUP,
    EVENT_THREAD_EXIT,
};

struct event {
    __u64 ts_ns;
    __u64 lock_addr;
    __s64 ret;
    __u32 pid;
    __u32 tid;
    __u32 target_tid;
    __u32 cpu;
    __u32 type;
    __u32 operation;
};

#endif
