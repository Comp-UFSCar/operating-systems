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
    int done,
        password_found,
        n_threads;
    char *filename;
    char *command_template;
} ck_st_t;


ck_st_t cracking_state;
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

    password = rank;

    while (!cracking_state.done && password < 500000)
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

        // Cyclical scheduling.
        password += cracking_state.n_threads;
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
    }

    for (i = 0; i < n_threads; i++) pthread_join(workers[i], NULL);

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

    workers = (pthread_t *) malloc(n_threads * sizeof(pthread_t));

    t_start = rtclock();
    int password_found = crack_password(filename, n_threads);
    t_end   = rtclock();

    printf("Senha:%d\n", password_found);
    fprintf(stdout, "%0.6lf\n", t_end - t_start);

    free(workers);
    return 0;
}

/*
## arq1.in
Senha:10000

### Time elapsed
* serial: 16.141632
* parallel: 7.850181

## arq2.in
Senha:100000

### Time elapsed
* serial: 160.687960
* parallel: 80.933334

## arq3.in
Senha:450000

## Time elapsed
* serial: 764.640047
* parallel: 350.083878

## arq4.in
Senha:310000

## Time elapsed
* serial: 538.292841
* parallel: 247.497154
*/
