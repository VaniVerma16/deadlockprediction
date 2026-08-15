#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "../events.h"

char LICENSE[] SEC("license") = "GPL";
const volatile __u32 target_tgid = 0;

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24);
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key, __u64);
    __type(value, __u64);
} pending_mutex SEC(".maps");

struct futex_pending {
    __u64 address;
    __u32 operation;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 16384);
    __type(key, __u64);
    __type(value, struct futex_pending);
} pending_futex SEC(".maps");

static __always_inline int is_target(__u64 pid_tgid) {
    return target_tgid == 0 || (__u32)(pid_tgid >> 32) == target_tgid;
}

static __always_inline void submit_event(__u32 type, __u64 address, __s64 ret,
                                         __u32 operation, __u32 target_tid) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    if (!is_target(pid_tgid)) return;

    struct event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) return;
    event->ts_ns = bpf_ktime_get_ns();
    event->lock_addr = address;
    event->ret = ret;
    event->pid = pid_tgid >> 32;
    event->tid = (__u32)pid_tgid;
    event->target_tid = target_tid;
    event->cpu = bpf_get_smp_processor_id();
    event->type = type;
    event->operation = operation;
    bpf_ringbuf_submit(event, 0);
}

SEC("uprobe")
int on_mutex_lock_enter(struct pt_regs *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    if (!is_target(key)) return 0;
    __u64 address = PT_REGS_PARM1_CORE(ctx);
    bpf_map_update_elem(&pending_mutex, &key, &address, BPF_ANY);
    submit_event(EVENT_LOCK_ATTEMPT, address, 0, 0, 0);
    return 0;
}

SEC("uretprobe")
int on_mutex_lock_exit(struct pt_regs *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    __u64 *address = bpf_map_lookup_elem(&pending_mutex, &key);
    if (!address) return 0;
    __s64 ret = PT_REGS_RC_CORE(ctx);
    if (ret == 0) submit_event(EVENT_LOCK_ACQUIRED, *address, ret, 0, 0);
    bpf_map_delete_elem(&pending_mutex, &key);
    return 0;
}

SEC("uprobe")
int on_mutex_trylock_enter(struct pt_regs *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    if (!is_target(key)) return 0;
    __u64 address = PT_REGS_PARM1_CORE(ctx);
    bpf_map_update_elem(&pending_mutex, &key, &address, BPF_ANY);
    submit_event(EVENT_TRYLOCK_ATTEMPT, address, 0, 0, 0);
    return 0;
}

SEC("uretprobe")
int on_mutex_trylock_exit(struct pt_regs *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    __u64 *address = bpf_map_lookup_elem(&pending_mutex, &key);
    if (!address) return 0;
    __s64 ret = PT_REGS_RC_CORE(ctx);
    if (ret == 0) submit_event(EVENT_TRYLOCK_ACQUIRED, *address, ret, 0, 0);
    bpf_map_delete_elem(&pending_mutex, &key);
    return 0;
}

SEC("uprobe")
int on_mutex_unlock_enter(struct pt_regs *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    if (!is_target(key)) return 0;
    submit_event(EVENT_LOCK_RELEASED, PT_REGS_PARM1_CORE(ctx), 0, 0, 0);
    return 0;
}

SEC("tracepoint/syscalls/sys_enter_futex")
int on_futex_enter(struct trace_event_raw_sys_enter *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    if (!is_target(key)) return 0;
    __u32 operation = (__u32)ctx->args[1];
    __u32 command = operation & 0x7f;
    if (command != 0 && command != 9) return 0; /* FUTEX_WAIT / FUTEX_WAIT_BITSET */
    struct futex_pending value = {
        .address = (__u64)ctx->args[0],
        .operation = operation,
    };
    bpf_map_update_elem(&pending_futex, &key, &value, BPF_ANY);
    submit_event(EVENT_FUTEX_WAIT, value.address, 0, operation, 0);
    return 0;
}

SEC("tracepoint/syscalls/sys_exit_futex")
int on_futex_exit(struct trace_event_raw_sys_exit *ctx) {
    __u64 key = bpf_get_current_pid_tgid();
    struct futex_pending *value = bpf_map_lookup_elem(&pending_futex, &key);
    if (!value) return 0;
    submit_event(EVENT_FUTEX_RETURN, value->address, ctx->ret, value->operation, 0);
    bpf_map_delete_elem(&pending_futex, &key);
    return 0;
}

SEC("tracepoint/sched/sched_switch")
int on_sched_switch(struct trace_event_raw_sched_switch *ctx) {
    submit_event(EVENT_SCHED_SWITCH, 0, ctx->prev_state, 0, ctx->next_pid);
    return 0;
}

SEC("tracepoint/sched/sched_wakeup")
int on_sched_wakeup(struct trace_event_raw_sched_wakeup_template *ctx) {
    submit_event(EVENT_THREAD_WAKEUP, 0, 0, 0, ctx->pid);
    return 0;
}

SEC("tracepoint/sched/sched_process_exit")
int on_process_exit(void *ctx) {
    (void)ctx;
    submit_event(EVENT_THREAD_EXIT, 0, 0, 0, 0);
    return 0;
}

