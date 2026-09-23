#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

pthread_mutex_t lock_a;
pthread_mutex_t lock_b;

typedef struct {
    int mode;
} ThreadArgs;


/* ---------------------------------------------------------
   SAFE MODE
   Threads use the same mutex, but release it normally.
   No deadlock.
   --------------------------------------------------------- */
void *safe_worker(void *arg)
{
    (void)arg;

    for (int i = 0; i < 20; i++) {
        pthread_mutex_lock(&lock_a);

        printf("SAFE: Thread %lu acquired Lock A\n",
               (unsigned long)pthread_self());

        usleep(100000);

        pthread_mutex_unlock(&lock_a);

        usleep(100000);
    }

    return NULL;
}


/* ---------------------------------------------------------
   CONTENTION MODE
   Both threads repeatedly compete for the same mutex.
   They wait for each other temporarily, but no deadlock.
   --------------------------------------------------------- */
void *contention_worker(void *arg)
{
    (void)arg;

    for (int i = 0; i < 30; i++) {
        pthread_mutex_lock(&lock_a);

        printf("CONTENTION: Thread %lu acquired Lock A\n",
               (unsigned long)pthread_self());

        usleep(200000);

        pthread_mutex_unlock(&lock_a);

        usleep(50000);
    }

    return NULL;
}


/* ---------------------------------------------------------
   DEADLOCK MODE
   Thread 1:
       Lock A -> waits for Lock B

   Thread 2:
       Lock B -> waits for Lock A

   This creates a circular wait.
   --------------------------------------------------------- */

void *deadlock_worker_a(void *arg)
{
    (void)arg;

    pthread_mutex_lock(&lock_a);

    printf("DEADLOCK: Thread A acquired Lock A\n");
    fflush(stdout);

    /*
     * Give Thread B enough time to acquire Lock B.
     */
    usleep(500000);

    printf("DEADLOCK: Thread A waiting for Lock B\n");
    fflush(stdout);

    pthread_mutex_lock(&lock_b);

    /*
     * This line should never be reached.
     */
    printf("DEADLOCK: Thread A acquired Lock B\n");
    fflush(stdout);

    pthread_mutex_unlock(&lock_b);
    pthread_mutex_unlock(&lock_a);

    return NULL;
}


void *deadlock_worker_b(void *arg)
{
    (void)arg;

    pthread_mutex_lock(&lock_b);

    printf("DEADLOCK: Thread B acquired Lock B\n");
    fflush(stdout);

    /*
     * Give Thread A enough time to acquire Lock A.
     */
    usleep(500000);

    printf("DEADLOCK: Thread B waiting for Lock A\n");
    fflush(stdout);

    pthread_mutex_lock(&lock_a);

    /*
     * This line should never be reached.
     */
    printf("DEADLOCK: Thread B acquired Lock A\n");
    fflush(stdout);

    pthread_mutex_unlock(&lock_a);
    pthread_mutex_unlock(&lock_b);

    return NULL;
}


/* ---------------------------------------------------------
   MAIN
   --------------------------------------------------------- */

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <safe|contention|deadlock>\n", argv[0]);
        return 1;
    }

    const char *mode = argv[1];

    pthread_t thread_a;
    pthread_t thread_b;

    /*
     * Initialize POSIX mutexes.
     */
    pthread_mutex_init(&lock_a, NULL);
    pthread_mutex_init(&lock_b, NULL);

    printf("========================================\n");
    printf("Native POSIX pthread workload\n");
    printf("PID: %d\n", getpid());
    printf("Mode: %s\n", mode);
    printf("========================================\n");
    fflush(stdout);


    /* ---------------- SAFE ---------------- */

    if (strcmp(mode, "safe") == 0) {

        pthread_create(&thread_a, NULL, safe_worker, NULL);
        pthread_create(&thread_b, NULL, safe_worker, NULL);

        pthread_join(thread_a, NULL);
        pthread_join(thread_b, NULL);
    }


    /* -------------- CONTENTION -------------- */

    else if (strcmp(mode, "contention") == 0) {

        pthread_create(&thread_a, NULL, contention_worker, NULL);
        pthread_create(&thread_b, NULL, contention_worker, NULL);

        pthread_join(thread_a, NULL);
        pthread_join(thread_b, NULL);
    }


    /* --------------- DEADLOCK --------------- */

    else if (strcmp(mode, "deadlock") == 0) {

        pthread_create(&thread_a, NULL, deadlock_worker_a, NULL);
        pthread_create(&thread_b, NULL, deadlock_worker_b, NULL);

        /*
         * The threads intentionally never finish because
         * they are deadlocked.
         *
         * Keep the main process alive so that the eBPF
         * monitor can continue observing it.
         */
        pthread_join(thread_a, NULL);
        pthread_join(thread_b, NULL);
    }


    else {
        printf("Unknown mode: %s\n", mode);
        printf("Use: safe, contention, or deadlock\n");
        return 1;
    }


    pthread_mutex_destroy(&lock_a);
    pthread_mutex_destroy(&lock_b);

    return 0;
}