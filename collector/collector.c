#define _GNU_SOURCE

#include <bpf/libbpf.h>
#include <dlfcn.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "deadlock.skel.h"
#include "events.h"

static volatile sig_atomic_t stopping;
static const char *run_id;
static FILE *output;

static void stop_handler(int signal_number) {
    (void)signal_number;
    stopping = 1;
}

static const char *event_name(__u32 type) {
    switch (type) {
        case EVENT_LOCK_ATTEMPT: return "lock_attempt";
        case EVENT_LOCK_ACQUIRED: return "lock_acquired";
        case EVENT_TRYLOCK_ATTEMPT: return "trylock_attempt";
        case EVENT_TRYLOCK_ACQUIRED: return "trylock_acquired";
        case EVENT_LOCK_RELEASED: return "lock_released";
        case EVENT_FUTEX_WAIT: return "futex_wait";
        case EVENT_FUTEX_RETURN: return "futex_return";
        case EVENT_SCHED_SWITCH: return "sched_switch";
        case EVENT_THREAD_WAKEUP: return "thread_wakeup";
        case EVENT_THREAD_EXIT: return "thread_exit";
        default: return "unknown";
    }
}

static int handle_event(void *context, void *data, size_t size) {
    (void)context;
    if (size < sizeof(struct event)) return 0;
    const struct event *event = data;
    fprintf(output,
            "{\"run_id\":\"%s\",\"ts_ns\":%llu,\"source\":\"ebpf\","
            "\"event\":\"%s\",\"pid\":%u,\"tid\":%u,\"cpu\":%u,"
            "\"lock_addr\":\"0x%llx\",\"ret\":%lld,\"operation\":%u,"
            "\"target_tid\":%u}\n",
            run_id, (unsigned long long)event->ts_ns, event_name(event->type),
            event->pid, event->tid, event->cpu,
            (unsigned long long)event->lock_addr, (long long)event->ret,
            event->operation, event->target_tid);
    fflush(output);
    return 0;
}

static long symbol_offset(const char *library, const char *symbol) {
    void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "dlopen %s: %s\n", library, dlerror());
        return -1;
    }
    void *address = dlsym(handle, symbol);
    Dl_info info;
    if (!address || dladdr(address, &info) == 0) {
        fprintf(stderr, "cannot resolve %s in %s\n", symbol, library);
        dlclose(handle);
        return -1;
    }
    long offset = (char *)address - (char *)info.dli_fbase;
    dlclose(handle);
    return offset;
}

static struct bpf_link *attach_uprobe(struct bpf_program *program, bool retprobe,
                                      pid_t pid, const char *library, const char *symbol) {
    long offset = symbol_offset(library, symbol);
    if (offset < 0) return NULL;
    return bpf_program__attach_uprobe(program, retprobe, pid, library, (size_t)offset);
}

static void usage(const char *program) {
    fprintf(stderr, "usage: %s --pid PID --run-id ID --libc PATH --output FILE\n", program);
}

int main(int argc, char **argv) {
    pid_t pid = 0;
    const char *libc_path = NULL;
    const char *output_path = NULL;
    run_id = NULL;
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--pid") == 0 && index + 1 < argc) pid = atoi(argv[++index]);
        else if (strcmp(argv[index], "--run-id") == 0 && index + 1 < argc) run_id = argv[++index];
        else if (strcmp(argv[index], "--libc") == 0 && index + 1 < argc) libc_path = argv[++index];
        else if (strcmp(argv[index], "--output") == 0 && index + 1 < argc) output_path = argv[++index];
        else { usage(argv[0]); return 2; }
    }
    if (pid <= 0 || !run_id || !libc_path || !output_path) { usage(argv[0]); return 2; }

    output = fopen(output_path, "w");
    if (!output) { perror("fopen output"); return 1; }
    signal(SIGINT, stop_handler);
    signal(SIGTERM, stop_handler);
    libbpf_set_strict_mode(LIBBPF_STRICT_ALL);

    struct deadlock_bpf *skeleton = deadlock_bpf__open();
    if (!skeleton) { fprintf(stderr, "failed to open BPF skeleton\n"); return 1; }
    skeleton->rodata->target_tgid = (__u32)pid;
    int error = deadlock_bpf__load(skeleton);
    if (error) { fprintf(stderr, "failed to load BPF programs: %d\n", error); goto cleanup; }

    struct bpf_link *links[10] = {0};
    links[0] = attach_uprobe(skeleton->progs.on_mutex_lock_enter, false, pid, libc_path, "pthread_mutex_lock");
    links[1] = attach_uprobe(skeleton->progs.on_mutex_lock_exit, true, pid, libc_path, "pthread_mutex_lock");
    links[2] = attach_uprobe(skeleton->progs.on_mutex_trylock_enter, false, pid, libc_path, "pthread_mutex_trylock");
    links[3] = attach_uprobe(skeleton->progs.on_mutex_trylock_exit, true, pid, libc_path, "pthread_mutex_trylock");
    links[4] = attach_uprobe(skeleton->progs.on_mutex_unlock_enter, false, pid, libc_path, "pthread_mutex_unlock");
    links[5] = bpf_program__attach_tracepoint(skeleton->progs.on_futex_enter, "syscalls", "sys_enter_futex");
    links[6] = bpf_program__attach_tracepoint(skeleton->progs.on_futex_exit, "syscalls", "sys_exit_futex");
    links[7] = bpf_program__attach_tracepoint(skeleton->progs.on_sched_switch, "sched", "sched_switch");
    links[8] = bpf_program__attach_tracepoint(skeleton->progs.on_sched_wakeup, "sched", "sched_wakeup");
    links[9] = bpf_program__attach_tracepoint(skeleton->progs.on_process_exit, "sched", "sched_process_exit");
    for (size_t index = 0; index < sizeof(links) / sizeof(links[0]); index++) {
        if (!links[index] || libbpf_get_error(links[index])) {
            fprintf(stderr, "failed to attach probe %zu\n", index);
            error = 1;
            goto links_cleanup;
        }
    }

    struct ring_buffer *ring = ring_buffer__new(bpf_map__fd(skeleton->maps.events), handle_event, NULL, NULL);
    if (!ring) { error = -errno; goto links_cleanup; }
    while (!stopping) {
        error = ring_buffer__poll(ring, 100);
        if (error == -EINTR) { error = 0; break; }
        if (error < 0) break;
    }
    ring_buffer__free(ring);

links_cleanup:
    for (size_t index = 0; index < sizeof(links) / sizeof(links[0]); index++) bpf_link__destroy(links[index]);
cleanup:
    deadlock_bpf__destroy(skeleton);
    fclose(output);
    return error < 0 ? -error : error;
}
