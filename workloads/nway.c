#include "common.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    run_config_t *config;
    int worker;
    unsigned int count;
    pthread_mutex_t *locks;
    sync_barrier_t *barrier;
} worker_args_t;

static void *worker_main(void *opaque) {
    worker_args_t *args = opaque;
    unsigned int own = (unsigned int)args->worker;
    unsigned int next = (own + 1) % args->count;
    unsigned int first = args->config->deadlock_mode || own < next ? own : next;
    unsigned int second = args->config->deadlock_mode || own < next ? next : own;
    unsigned int rounds = args->config->deadlock_mode ? 1 : args->config->iterations;

    for (unsigned int round = 0; round < rounds; round++) {
        if (!args->config->deadlock_mode) barrier_wait(args->barrier);
        emit_event(args->config, args->worker, "ground_truth_lock_attempt", &args->locks[first], NULL);
        pthread_mutex_lock(&args->locks[first]);
        emit_event(args->config, args->worker, "ground_truth_lock_acquired", &args->locks[first], NULL);
        if (args->config->deadlock_mode) barrier_wait(args->barrier);
        if (args->config->deadlock_mode) {
            unsigned int spacing = args->count > 1 ? 450 / (args->count - 1) : 0;
            sleep_ms(25 + spacing * (unsigned int)args->worker + args->config->seed % 15);
        } else {
            sleep_ms(2 + (args->config->seed + (unsigned int)args->worker + round) % 5);
        }
        emit_event(args->config, args->worker, "ground_truth_lock_attempt", &args->locks[second], NULL);
        pthread_mutex_lock(&args->locks[second]);
        emit_event(args->config, args->worker, "ground_truth_lock_acquired", &args->locks[second], NULL);
        sleep_ms(1 + (round + (unsigned int)args->worker) % 3);
        pthread_mutex_unlock(&args->locks[second]);
        pthread_mutex_unlock(&args->locks[first]);
        if (!args->config->deadlock_mode) {
            barrier_wait(args->barrier);
            sleep_ms(1 + (args->config->seed + round) % 3);
        }
    }
    emit_event(args->config, args->worker, "worker_complete", NULL, NULL);
    return NULL;
}

int main(int argc, char **argv) {
    run_config_t config = parse_config(argc, argv, "nway", 4);
    pthread_mutex_t *locks = calloc(config.threads, sizeof(*locks));
    pthread_t *threads = calloc(config.threads, sizeof(*threads));
    worker_args_t *args = calloc(config.threads, sizeof(*args));
    sync_barrier_t barrier;
    if (!locks || !threads || !args) return 2;

    for (unsigned int i = 0; i < config.threads; i++) pthread_mutex_init(&locks[i], NULL);
    barrier_init(&barrier, config.threads);
    emit_event(&config, -1, "run_start", NULL, NULL);
    sleep_ms(config.start_delay_ms);
    start_watchdog(&config);

    for (unsigned int i = 0; i < config.threads; i++) {
        args[i] = (worker_args_t){&config, (int)i, config.threads, locks, &barrier};
        pthread_create(&threads[i], NULL, worker_main, &args[i]);
    }
    for (unsigned int i = 0; i < config.threads; i++) pthread_join(threads[i], NULL);
    emit_event(&config, -1, "run_end", NULL, "completed");
    barrier_destroy(&barrier);
    free(args);
    free(threads);
    free(locks);
    return 0;
}
