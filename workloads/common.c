#define _POSIX_C_SOURCE 200809L

#include "common.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static run_config_t watchdog_config;
static pthread_mutex_t output_mutex = PTHREAD_MUTEX_INITIALIZER;

uint64_t monotonic_ns(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

void emit_event(const run_config_t *config, int worker, const char *event,
                const pthread_mutex_t *lock, const char *value) {
    pthread_mutex_lock(&output_mutex);
    printf("{\"run_id\":\"%s\",\"ts_ns\":%llu,\"source\":\"workload\","
           "\"event\":\"%s\",\"scenario\":\"%s\",\"mode\":\"%s\","
           "\"seed\":%u,\"pid\":%d,\"worker\":%d",
           config->run_id, (unsigned long long)monotonic_ns(), event,
           config->scenario, config->deadlock_mode ? "deadlock" : "safe",
           config->seed, (int)getpid(), worker);
    if (lock != NULL) {
        printf(",\"lock_addr\":\"%p\"", (const void *)lock);
    }
    if (value != NULL) {
        printf(",\"value\":\"%s\"", value);
    }
    puts("}");
    fflush(stdout);
    pthread_mutex_unlock(&output_mutex);
}

void sleep_ms(unsigned int milliseconds) {
    struct timespec duration = {
        .tv_sec = milliseconds / 1000,
        .tv_nsec = (long)(milliseconds % 1000) * 1000000L,
    };
    while (nanosleep(&duration, &duration) == -1 && errno == EINTR) {
    }
}

static void *watchdog_main(void *unused) {
    (void)unused;
    sleep_ms(watchdog_config.timeout_ms);
    emit_event(&watchdog_config, -1, "run_end", NULL, "timeout");
    _exit(watchdog_config.deadlock_mode ? 0 : 2);
}

void start_watchdog(const run_config_t *config) {
    pthread_t thread;
    watchdog_config = *config;
    if (pthread_create(&thread, NULL, watchdog_main, NULL) != 0) {
        perror("pthread_create watchdog");
        exit(2);
    }
    pthread_detach(thread);
}

static unsigned int parse_uint(const char *name, const char *value) {
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (value[0] == '\0' || end == NULL || *end != '\0' || parsed > 1000000UL) {
        fprintf(stderr, "invalid %s: %s\n", name, value);
        exit(2);
    }
    return (unsigned int)parsed;
}

run_config_t parse_config(int argc, char **argv, const char *scenario, unsigned int default_threads) {
    run_config_t config = {
        .run_id = "manual-run",
        .scenario = scenario,
        .deadlock_mode = 1,
        .seed = 1,
        .timeout_ms = 1000,
        .start_delay_ms = 500,
        .threads = default_threads,
        .iterations = 32,
    };
    for (int index = 1; index < argc; index++) {
        if (strcmp(argv[index], "--mode") == 0 && index + 1 < argc) {
            const char *mode = argv[++index];
            if (strcmp(mode, "deadlock") == 0) config.deadlock_mode = 1;
            else if (strcmp(mode, "safe") == 0) config.deadlock_mode = 0;
            else {
                fprintf(stderr, "mode must be safe or deadlock\n");
                exit(2);
            }
        } else if (strcmp(argv[index], "--run-id") == 0 && index + 1 < argc) {
            config.run_id = argv[++index];
        } else if (strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
            config.seed = parse_uint("seed", argv[++index]);
        } else if (strcmp(argv[index], "--timeout-ms") == 0 && index + 1 < argc) {
            config.timeout_ms = parse_uint("timeout-ms", argv[++index]);
        } else if (strcmp(argv[index], "--threads") == 0 && index + 1 < argc) {
            config.threads = parse_uint("threads", argv[++index]);
        } else if (strcmp(argv[index], "--start-delay-ms") == 0 && index + 1 < argc) {
            config.start_delay_ms = parse_uint("start-delay-ms", argv[++index]);
        } else if (strcmp(argv[index], "--iterations") == 0 && index + 1 < argc) {
            config.iterations = parse_uint("iterations", argv[++index]);
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[index]);
            exit(2);
        }
    }
    if (config.threads < 2 || config.threads > 64 || config.timeout_ms < 50
            || config.iterations < 1 || config.iterations > 10000) {
        fprintf(stderr, "threads must be 2..64, timeout-ms at least 50, and iterations 1..10000\n");
        exit(2);
    }
    return config;
}

void barrier_init(sync_barrier_t *barrier, unsigned int target) {
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->condition, NULL);
    barrier->arrived = 0;
    barrier->target = target;
    barrier->generation = 0;
}

void barrier_wait(sync_barrier_t *barrier) {
    pthread_mutex_lock(&barrier->mutex);
    unsigned int generation = barrier->generation;
    barrier->arrived++;
    if (barrier->arrived == barrier->target) {
        barrier->arrived = 0;
        barrier->generation++;
        pthread_cond_broadcast(&barrier->condition);
    } else {
        while (generation == barrier->generation) {
            pthread_cond_wait(&barrier->condition, &barrier->mutex);
        }
    }
    pthread_mutex_unlock(&barrier->mutex);
}

void barrier_destroy(sync_barrier_t *barrier) {
    pthread_cond_destroy(&barrier->condition);
    pthread_mutex_destroy(&barrier->mutex);
}
