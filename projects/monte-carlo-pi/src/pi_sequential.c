#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>


long long unsigned int
monte_carlo_pi (unsigned int n_epochs)
{
	long long unsigned int in = 0, i;
	double x, y, d;

	for (i = 0; i < n_epochs; i++)
	{
		x = (rand() % 1000000) / 500000.0 - 1;
		y = (rand() % 1000000) / 500000.0 - 1;
		d = x * x + y * y;

		if (d <= 1) in += 1;
	}

	return in;
}


double
pi (int n_epochs)
{
    
    return 4 * monte_carlo_pi(n_epochs) / (double) n_epochs;
}


int
main (void)
{
	int n_threads;
	unsigned int n_epochs;
	long unsigned int elapsed;
	struct timeval start, end;

	scanf("%d %u", &n_threads, &n_epochs);
	srand (time(NULL));

	gettimeofday(&start, NULL);
	double estimated_pi = pi(n_epochs);
	gettimeofday(&end, NULL);

	elapsed = end.tv_sec * 1000000 + end.tv_usec
	           - start.tv_sec * 1000000 + start.tv_usec;

	printf("%lf\n%lu\n", estimated_pi, elapsed);

	return 0;
}
