#ifndef DEADLOCK_COMMON_H
#define DEADLOCK_COMMON_H

#include <pthread.h>
#include <stdint.h>

typedef struct {
    const char *run_id;
    const char *scenario;
    int deadlock_mode;
    unsigned int seed;
    unsigned int timeout_ms;
    unsigned int start_delay_ms;
    unsigned int threads;
    unsigned int iterations;
} run_config_t;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned int arrived;
    unsigned int target;
    unsigned int generation;
} sync_barrier_t;

run_config_t parse_config(int argc, char **argv, const char *scenario, unsigned int default_threads);
uint64_t monotonic_ns(void);
void emit_event(const run_config_t *config, int worker, const char *event,
                const pthread_mutex_t *lock, const char *value);
void sleep_ms(unsigned int milliseconds);
void start_watchdog(const run_config_t *config);
void barrier_init(sync_barrier_t *barrier, unsigned int target);
void barrier_wait(sync_barrier_t *barrier);
void barrier_destroy(sync_barrier_t *barrier);

#endif
