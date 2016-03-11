#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>


double *
count_sort_parallel(double sequence[], int len, int n_threads) {
	int i, j, count;
	double *sorted = (double *)malloc(len * sizeof(double));

	// `j` and `count` should be individual for each thread.
	# pragma omp parallel for num_threads(n_threads) private(count, j)
	for (i = 0; i < len; i++) {
		count = 0;

		for (j = 0; j < len; j++)
			// Else statement was removed because it's a logic abomination.
			if (sequence[j] < sequence[i]
				|| (sequence[j] == sequence[i] && j < i))
				count++;

		sorted[count] = sequence[i];
	}

	return sorted;
}


int
main(int argc, char * argv[]) {
	int i, len, n_threads;

	scanf("%d", &n_threads);
	scanf("%d", &len);

	double *sequence = (double *)malloc(len*sizeof(double)),
		   *sorted;

	for(i = 0; i < len; i++)
		scanf("%lf", &sequence[i]);

	double time_elapsed =  omp_get_wtime();
	sorted = count_sort_parallel(sequence, len, n_threads);
	time_elapsed = omp_get_wtime() - time_elapsed;

	for(i = 0; i < len; i++)
		printf("%.2lf ", sorted[i]);

	printf("\n");
	printf("%lf\n", time_elapsed);

	return 0;
}
