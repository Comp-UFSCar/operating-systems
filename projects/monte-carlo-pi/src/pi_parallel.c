#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<sys/time.h>


typedef struct {
    unsigned int seed;
    int n_epochs;
    long long int in;
} pi_params_t;


void
pi_computer (void *t_params)
{
    long long unsigned int i;
	double x, y, d;

    pi_params_t *params = (pi_params_t *) t_params;
	params->in = 0;

	for (i = 0; i < params->n_epochs; i++)
	{
		x = (rand_r(&params->seed) % 1000000) / 500000.0 - 1;
		y = (rand_r(&params->seed) % 1000000) / 500000.0 - 1;
		d = x * x + y * y;

		if (d <= 1) params->in += 1;
	}
}


double
pi (int n_epochs, int n_threads)
{
    int i, failed;
    long long int global_in = 0;
    
    pthread_t   *workers = (pthread_t *)  malloc(n_threads * sizeof(pthread_t));
    pi_params_t *params  = (pi_params_t *)malloc(n_threads * sizeof(pi_params_t));
    
    for (i = 0; i < n_threads; i++)
    {
        // Every worker has their own random seed.
        params[i].seed = rand();
        params[i].n_epochs = n_epochs / n_threads;
        
        if (i == n_threads -1)
            params[i].n_epochs += n_epochs % n_threads;
        
        failed = pthread_create(&workers[i], NULL, pi_computer, (void *)&params[i]);
        
        if (failed)
        {
            fprintf(stderr, "Failed to create thread (code: %d).\n", failed);
            exit(EXIT_FAILURE);
        }
    }
    
    for (i = 0; i < n_threads; i++)
    {
        pthread_join(workers[i], NULL);
        global_in += params[i].in;
    }
    
    free(workers);
    free(params);

    return 4 * global_in / (double) n_epochs;
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
