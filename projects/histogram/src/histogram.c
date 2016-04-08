//
// histogram.c
//
// Multithreading solution for bins filling.
//
// @author Lucas David - ld492@drexel.edu
//

#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>


typedef struct {
    int rank, *bins, n_values, n_bins, n_threads;
    double *values, min, max, h;
} histogram_params_t;


//
// Min.
//
// Find the minimum between two values.
//
// Returns
// -------
// int : the minimum value.
//
double min(double a, double b) { return a <= b ? a : b; }


//
// Sequential Histogram.
//
// Compute an histogram based on values and the bins passed.
//
void
sequential_histogram(double *values, int n_values, int *bins, int n_bins,
                     double min, double max, double h)
{
    int i, j, count;
    double min_t, max_t;

    for (j = 0; j < n_bins; j++)
    {
        count = 0;
        min_t = min + j * h;
		max_t = min + (j+1) * h;

		for(i = 0; i < n_values; i++)
			if(values[i] > min_t && values[i] <= max_t) count++;

		bins[j] = count;
	}
}


//
// Histogram Worker.
//
// Parallel worker for computing bins' values.
//
// Parameters
// ----------
// thread_params : histogram_params_t
//     Parameters used for this worker, such as rank,
//     pointers and reference counters.
//
void histogram_worker(void *thread_params)
{
    int i, j, count;
    double min_t, max_t;
    histogram_params_t *params = (histogram_params_t *)thread_params;

    // `ceil` and double division is used to distribute `n_bins % n_threads`
    // equally between the threads.
    // Although the last thread may get fewer bins to compute, this is still
    // a better strategy than giving `n_bins % n_threads` to the last thread.
    int start = ceil(params->rank
                     * (double) params->n_bins / params->n_threads),
        end   = min(params->n_bins,
                    ceil((params->rank + 1)
                         * (double)params->n_bins / params->n_threads));

    for (j = start; j < end; j++)
    {
        count = 0;
        min_t = params->min + j * params->h;
		max_t = params->min + (j + 1) * params->h;

		for (i = 0; i < params->n_values; i++)
			if (params->values[i] > min_t && params->values[i] <= max_t)
                count++;

		params->bins[j] = count;
	}
}


//
// Parallel Histogram.
//
// Compute an histogram based on values and bins passed.
//
// Parameters
// ----------
// n_threads : int
//             number of threads workers (threads) used in the computation.
//
void
parallel_histogram(double *values, int n_values, int *bins, int n_bins,
                   double min, double max, double h, int n_threads)
{
    int i, failed;
    pthread_t *workers = (pthread_t *) malloc(n_threads * sizeof(pthread_t));
    histogram_params_t *params
        = (histogram_params_t *)malloc(n_threads * sizeof(histogram_params_t));

    for (i = 0; i < n_threads; i++)
    {
        params[i].rank = i;
        params[i].bins = bins;
        params[i].values = values;
        params[i].n_values = n_values;
        params[i].n_threads = n_threads;
        params[i].bins = bins;
        params[i].n_bins = n_bins;
        params[i].min = min;
        params[i].max = max;
        params[i].h = h;

        failed = pthread_create(&workers[i], NULL, histogram_worker,
                                (void *) &params[i]);

        if (failed)
        {
            fprintf(stderr, "Failed to create thread (code: %d).\n", failed);
            exit(EXIT_FAILURE);
        }
    }

    for (i = 0; i < n_threads; i++) pthread_join(workers[i], NULL);

    free(workers);
}


//
// Main.
//
int main(int argc, char * argv[]) {
	int *bins, n_threads, n_bins, n_values, i;
	double h, *values, max, min;
	long unsigned int elapsed;
	struct timeval start, end;

	scanf("%d", &n_threads);
	scanf("%d", &n_values);
	scanf("%d", &n_bins);

	values = (double *) malloc(n_values * sizeof(double));
	bins   = (int *)    malloc(n_bins   * sizeof(int));

	for(i = 0; i < n_values; i++) scanf("%lf",&values[i]);

    // Compute min and max values.
    min = max = values[0];

    for (i = 1; i < n_values; i++)
        if (values[i] < min) min = values[i];
        else if (values[i] > max) max = values[i];

    min = floor(min);
    max = ceil(max);
	h = (max - min) / n_bins;

	gettimeofday(&start, NULL);
	parallel_histogram(values, n_values, bins, n_bins, min, max, h, n_threads);

	gettimeofday(&end, NULL);

	elapsed = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf", min);
	for(i = 1; i <= n_bins; i++) printf(" %.2lf", min + h * i);
	printf("\n");

	printf("%d", bins[0]);
	for(i = 1; i < n_bins; i++) printf(" %d", bins[i]);
	printf("\n");
	printf("%lu\n", elapsed);

	free(bins);
	free(values);
	free(bounds);

	return 0;
}

/*
 * Complementary work - Speed Up and Efficiency.
 * ---------------------------------------------
 *
 * Time elapsed:
 * f1 : 1928
 * f2 : 52726
 * f3 : 399156
 *
 * Time elapsed (2 threads):
   f1 : 732
   f2 : 29064
   f3 : 202268
 *
 * Time elapsed (4 threads):
   f1 : 342
   f2 : 21965
   f3 : 163407
 *
 * Time elapsed (8 threads):
   f1 : 404
   f2 : 13829
   f3 : 94118
 *
 * Time elapsed (16 threads):
   f1 : 408
   f2 : 15684
   f3 : 82152
 *
 *           T | 1 | 2   | 4   | 8   | 16
 * arq1.in | S | 1 | 2.6 | 5.6 | 4.8 | 4.7
           | E | 1 | 1.3 | 1.4 | 0.6 | 0.3
 * arq2.in | S | 1 | 1.8 | 2.4 | 3.8 | 3.4
           | E | 1 | 0.9 | 0.6 | 0.5 | 0.2
 * arq3.in | S | 1 | 2   | 2.4 | 4.2 | 4.9
           | E | 1 | 1   | 0.6 | 0.5 | 0.3
 *
 * Comments on speedup and efficiency
 * ------------
 *
 * arq1.in
 *    2 threads: efficiency was suprisingly higher than 1: 1.3. My guess is
                 that cache hit has maintained itself consistently high.
 *    4 threads: efficiency stood by 1.4, being the most efficient option.
                 Said setting matches my computer phisical CPU count.
                 Furtheremore, there's few data (relative to the other files).
                 This two facts together are likely culprits for 4 threads
                 outperforming the other tests.
 *    8 threads: speedup and efficiency have actually dropped, in comparison
                 to 4 threads. This is likely because arq1.in doesn't contain
                 a great enough sequence that compensates for the work
                 necessary in managing the threads.
 *   16 threads: once again, speedup and efficiency have dropped. This was
                 expected considering my computer has 8 cores, and so
                 much of the time was spent on context switching instead
                 of actual bin computations.
 *
 * arq2.in
 *    2 threads: efficiency was almost 1, hence the speedup was almost linear.
 *    4 threads: the speed up was 2.4, an increase of .6 when compared to the
                 execution with 2 threads. Efficiency, in the other hand, drops
                 to .6.
 *    8 threads: 3.8 of speedup, being the higher of this group.
                 More importantly efficiency only dropped .1, when compared
                 to 4 threads!
 *   16 threads: Both speedup and efficiency have dropped. With .2 of effiency,
                 it's the smaller number in this group.
 *
 * arq3.in
 *    2 threads: linear speedup! The data contained in arq3.in seems to be
                 ideally divided between two groups.
 *    4 threads: both speedup and efficiency maintained close to the results
                 found with arg2.in: 2.4 of speedup and .6 of efficiency.
 *    8 threads: 4.2 of speedup, 2.0 more than 4 threads and 2.2 more
                 than 2 threads, representing a consistent speedup.
                 Sadly, efficiency is at .5.
 *   16 threads: We have the greatest speedup of this group: 4.9. This is also
                 the only data in which 16 threads outperformed 8 threads,
                 which is awkward considering this program doesn't
                 intertwine IO and actual computation. My only guess is that
                 I was unfortuned when running the 8 thread test and some of
                 the cores were busy with other tasks and reduced its
                 speedup.
 */
