#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include<pthread.h>
#include <sys/time.h>

#define CANCEL_CRACKING_PROCESS -10

FILE *popen(const char *command, const char *type);


typedef struct {
    void **items;
    int capacity;
    int count;
} queue_t;

queue_t *
queue_new(int capacity)
{
    queue_t *new = (queue_t *) malloc(sizeof(queue_t));

    new->items = malloc(capacity * sizeof(void *));
    new->capacity = capacity;

    return new;
}

void
queue_delete(queue_t *queue)
{
    free(queue->items);
    free(queue);
}

int
queue_add(queue_t *queue, void *item)
{
    if (queue->count == queue->capacity)
        return 0;

    queue->items[queue->count] = item;
    queue->count++;
}

void *
queue_pop(queue_t *queue)
{
    if (queue->count == 0)
        return NULL;

    void *item = queue->items[queue->count -1];
    queue->count--;

    return item;
}


typedef struct {
    int start,
        end;
} chunk_t;

///
/// Cracking State Type.
///
/// Contains shared info for all threads working
/// on cracking the zip file password.
///
typedef struct {
    int done,
        password_found,
        n_threads;
    char *filename;
    char *command_template;
} ck_st_t;


ck_st_t cracking_state;

queue_t **queues;
pthread_t *workers;

double
rtclock ()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void *
pass_cracker (void *t_params)
{
    FILE * fp;
    int rank = (int) t_params, password;
    char command[400], result[200];
    void *el;
    chunk_t *chunk;

    queue_t *queue = queues[rank];

    while (queue->count)
    {
        el = queue_pop(queue);

        if (el == CANCEL_CRACKING_PROCESS)
            // Cancel order was issued.
            // Notice we can safely compare `el` to `CANCEL_CRACKING_PROCESS`
            // because no valid memory block assumes the address `-10`.
            break;

        chunk = (chunk_t *) el;

        for (password = chunk->start; password < chunk->end; password++)
        {
            // For every password within the task range.
            // Build the command considering that password.
            sprintf((char*)&command, cracking_state.command_template,
                    password, cracking_state.filename);

            fp = popen(command, "r");
            while (!feof(fp))
            {
                fgets((char*)&result, 200, fp);
                if (strcasestr(result, "ok") != NULL)
                {
                    // Password is correct. Inform other threads.
                    cracking_state.password_found = password;
                    cracking_state.done = 1;
                }
            }

            pclose(fp);
        }
    }
}


///
/// Crack Password.
///
/// Parallel API wrapper.
///
int
crack_password (char* filename, int n_threads)
{
    int i, failed;

    int queue_size = 500000 / n_threads,
        chunk_size = .01 * 500000,  // 1% of total.
        n_chunks = 500000 / chunk_size;

    cracking_state.done = 0;
    cracking_state.password_found = -1;
    cracking_state.filename = filename;
    cracking_state.n_threads = n_threads;
    cracking_state.command_template = "unzip -P%d -t %s 2>&1";

    for (i = 0; i < n_threads; i++)
    {
        failed = pthread_create(&workers[i], NULL, pass_cracker, (void *)i);
        if (failed)
        {
            fprintf(stderr, "Failed to create thread (code: %d).\n", failed);
            exit(EXIT_FAILURE);
        }

        // Create each worker's queue.
        queues[i] = queue_new(queue_size);
    }

    int thread = 0;
    for (i = 0; i < n_chunks; i++)
    {
        // Create a chunk (an interval of the 500000 possible passwords).
        chunk_t *c = (chunk_t *) malloc(sizeof(chunk_t));

        c->start = (i)     * chunk_size;
        c->end   = (i + 1) * chunk_size;

        // Give the chunk to a thread.
        queue_add(queues[thread], c);

        // Move to the next thread, which has
        // fewer enqueued tasks than the previous.
        thread = (thread + 1) % n_threads;
    }

    printf("Checking if finished... ");
    while (!cracking_state.done)
        printf(" No.\n");
        // Sleep for a while to prevent high CPU usage.
        sleep(2000);
        printf("Checking if finished... ");
    printf("Yes.\n");

    // Password was found! Issue an order to abort search.
    // Notice we don't really join. We don't even care if they finished or not
    // before returning the password found.
    for (i = 0; i < n_threads; i++)
    {
        queue_add(queues[n_threads], CANCEL_CRACKING_PROCESS);
    }

    return cracking_state.password_found;
}


///
/// Main.
///
int
main ()
{
    int n_threads;
    char filename[100];
    double t_start, t_end;

    scanf("%d", &n_threads);
    scanf("%s", filename);

    queues = (queue_t *) malloc(n_threads * sizeof(queue_t *));
    workers = (pthread_t *) malloc(n_threads * sizeof(pthread_t));

    t_start = rtclock();
    int password_found = crack_password(filename, n_threads);
    t_end   = rtclock();

    printf("Senha:%d\n", password_found);
    fprintf(stdout, "%0.6lf\n", t_end - t_start);

    free(workers);
    return 0;
}
