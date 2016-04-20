#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include<pthread.h>
#include <sys/time.h>


FILE *popen(const char *command, const char *type);


///
/// Cracking State Type.
///
/// Contains shared info for all threads working
/// on cracking the zip file password.
///
typedef struct {
    int done;
    int password_found;
    char command_template[300];
} ck_st_t;

ck_st_t state;
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
crack_password_worker (void *t_params)
{
    int rank = (int) t_params;

    FILE * fp;
    char command[400], result[200];

    while (!state.done)
    {
        int password = next_password();
        sprintf((char*)&command, state.command_template, password, filename);

        fp = popen(command, "r");
        while (!feof(fp))
        {
            fgets((char*)&result, 200, fp);
            if (strcasestr(result, "ok") != NULL)
            {
                state.password_found = password;
                state.done = 1;

                printf("Senha:%d\n", password);
            }
        }
        pclose(fp);
    }
}


int
crack_password (char* filename, int n_threads) {
    int failed;

    state.done = 0;
    state.password_found = -1;
    state.command_template = "unzip -P%d -t %s 2>&1";

    for (i = 0; i < n_threads; i++)
    {
        failed = pthread_create(&workers[i], NULL, crack_password_worker,
                                (void *)&params[i]);
        if (failed)
        {
            fprintf(stderr, "Failed to create thread (code: %d).\n", failed);
            exit(EXIT_FAILURE);
        }
    }

    for (i = 0; i < n_threads; i++) pthread_join(workers[i], NULL);

    return state.password_found;
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

    workers = (pthread_t *) malloc(n_threads * sizeof(pthread_t));

    t_start = rtclock();
    crack_password(filename, n_threads);
    t_end = rtclock();

    fprintf(stdout, "%0.6lf\n", t_end - t_start);

    free(workers);
    return 0;
}
