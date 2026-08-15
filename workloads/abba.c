#include "common.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    run_config_t *config;
    int worker;
    pthread_mutex_t *first;
    pthread_mutex_t *second;
    sync_barrier_t *barrier;
} worker_args_t;

static void *worker_main(void *opaque) {
    worker_args_t *args = opaque;
    unsigned int rounds = args->config->deadlock_mode ? 1 : args->config->iterations;
    for (unsigned int round = 0; round < rounds; round++) {
        if (!args->config->deadlock_mode) barrier_wait(args->barrier);
        emit_event(args->config, args->worker, "ground_truth_lock_attempt", args->first, NULL);
        pthread_mutex_lock(args->first);
        emit_event(args->config, args->worker, "ground_truth_lock_acquired", args->first, NULL);
        if (args->config->deadlock_mode) barrier_wait(args->barrier);
        if (args->config->deadlock_mode) {
            sleep_ms(35 + 450 * (unsigned int)args->worker + args->config->seed % 20);
        } else {
            sleep_ms(3 + (args->config->seed + (unsigned int)args->worker + round) % 6);
        }
        emit_event(args->config, args->worker, "ground_truth_lock_attempt", args->second, NULL);
        pthread_mutex_lock(args->second);
        emit_event(args->config, args->worker, "ground_truth_lock_acquired", args->second, NULL);
        sleep_ms(1 + (round + (unsigned int)args->worker) % 3);
        pthread_mutex_unlock(args->second);
        pthread_mutex_unlock(args->first);
        if (!args->config->deadlock_mode) {
            barrier_wait(args->barrier);
            sleep_ms(2 + (args->config->seed + round) % 4);
        }
    }
    emit_event(args->config, args->worker, "worker_complete", NULL, NULL);
    return NULL;
}

int main(int argc, char **argv) {
    run_config_t config = parse_config(argc, argv, "abba", 2);
    if (config.threads != 2) {
        fprintf(stderr, "ABBA requires exactly 2 threads\n");
        return 2;
    }
    pthread_mutex_t lock_a = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t lock_b = PTHREAD_MUTEX_INITIALIZER;
    sync_barrier_t barrier;
    pthread_t threads[2];
    worker_args_t args[2];

    barrier_init(&barrier, 2);
    emit_event(&config, -1, "run_start", NULL, NULL);
    sleep_ms(config.start_delay_ms);
    start_watchdog(&config);

    args[0] = (worker_args_t){&config, 0, &lock_a, &lock_b, &barrier};
    args[1] = config.deadlock_mode
        ? (worker_args_t){&config, 1, &lock_b, &lock_a, &barrier}
        : (worker_args_t){&config, 1, &lock_a, &lock_b, &barrier};

    for (int i = 0; i < 2; i++) pthread_create(&threads[i], NULL, worker_main, &args[i]);
    for (int i = 0; i < 2; i++) pthread_join(threads[i], NULL);
    emit_event(&config, -1, "run_end", NULL, "completed");
    barrier_destroy(&barrier);
    return 0;
}
