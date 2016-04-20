#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<sys/time.h>


typedef struct {
    unsigned int seed;
    int n_epochs;
    long long int *in;
} pi_params_t;


pthread_mutex_t pi_update_lock;


void
pi_computer (void *t_params)
{
    pi_params_t *params = (pi_params_t *) t_params;

    long long unsigned int i;
	double x, y, d;
    long long int in = 0;
    int n_epochs = params->n_epochs;
    unsigned int seed = params->seed;

	for (i = 0; i < n_epochs; i++)
	{
		x = (rand_r(&seed) % 1000000) / 500000.0 - 1;
		y = (rand_r(&seed) % 1000000) / 500000.0 - 1;
		d = x * x + y * y;

		if (d <= 1) in += 1;
	}

    pthread_mutex_lock(&pi_update_lock);
    *params->in += in;
    pthread_mutex_unlock(&pi_update_lock);
}


double
pi (int n_epochs, int n_threads)
{
    int i, failed;
    long long int result = 0;

    pthread_t   *workers = (pthread_t *)  malloc(n_threads * sizeof(pthread_t));
    pi_params_t *params  = (pi_params_t *)malloc(n_threads * sizeof(pi_params_t));

    int n_local_epochs = n_epochs / n_threads;

    for (i = 0; i < n_threads; i++)
    {
        // Every worker has their own random seed.
        params[i].seed = rand();
        params[i].n_epochs = n_local_epochs;
        params[i].in = &result;

        if (i == n_threads -1) params[i].n_epochs += n_epochs % n_threads;

        failed = pthread_create(&workers[i], NULL, pi_computer,
                                (void *)&params[i]);
        if (failed)
        {
            fprintf(stderr, "Failed to create thread (code: %d).\n", failed);
            exit(EXIT_FAILURE);
        }
    }

    for (i = 0; i < n_threads; i++) pthread_join(workers[i], NULL);

    return 4 * result / (double) n_epochs;
}


int
main (void)
{
	int n_threads;
	unsigned int n_epochs;
	long unsigned int elapsed;
	struct timeval start, end;

	scanf("%d %u", &n_threads, &n_epochs);
	srand(time(NULL));

	gettimeofday(&start, NULL);
	double estimated_pi = pi(n_epochs, n_threads);
	gettimeofday(&end, NULL);

	elapsed = end.tv_sec * 1000000 + end.tv_usec
	           - start.tv_sec * 1000000 + start.tv_usec;

	printf("%lf\n%lu\n", estimated_pi, elapsed);

	return 0;
}
