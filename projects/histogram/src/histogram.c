#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


//
// min_max_in.
//
// Compute minimum and maximum values in a vector.
//
// Returns
// -------
// tuple (min, max), containing the minimum and maximum values found.
//
double *
min_max_in(double *vector, int length)
{
    int MIN = 0, MAX = 1, i;
    double *bounds = (double *) malloc(2 * sizeof(double));
    
    bounds[MIN] = bounds[MAX] = vector[0];
    
    for (i = 1; i < length; i++)
        if (vector[i] < bounds[MIN]) bounds[MIN] = vector[i];
        else if (vector[i] > bounds[MAX]) bounds[MAX] = vector[i];
    
    return bounds;
}


// bounds_to_integers.
//
// Maps real boundaries to integers set.
//
// Parameters
// ----------
// bound : tuple (min, max)
//         boundary that should be mapped.
//
// Returns
// -------
// tuple (min, max), where min and max are integers.
//
double *
bounds_to_integers(double *bounds)
{
    int MIN = 0, MAX = 1;
    
    bounds[MIN] = floor(bounds[MIN]);
    bounds[MAX] = floor(bounds[MAX]);
    
    return bounds;
}


// compute_histogram.
//
// Compute an histogram based on values and 
// the bins passed.
//
void
compute_histogram(double *values, int n_values, double min, double max, int *bins, int n_bins)
{
    int i, j, count;
    double min_t, max_t, h;
    
    h = (max - min) / n_bins;

    for (j = 0; j < n_bins; j++)
    {
        count = 0;
        min_t = min + j * h;
		max_t = min + (j+1) * h;
		
		for(i = 0; i < n_values; i++)
			if(values[i] <= max_t && values[i] > min_t) count++;

		bins[j] = count;
	}
}

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
	
	double *bounds = bounds_to_integers(min_max_in(values, n_values));
	
	min = bounds[0];
	max = bounds[1];
	h = (max - min) / n_bins;
	
	gettimeofday(&start, NULL);
	compute_histogram(values, n_values, min, max, bins, n_bins);
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
