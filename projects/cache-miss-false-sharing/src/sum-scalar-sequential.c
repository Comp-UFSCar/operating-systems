/* 
 * sum_scalar.c - A simple parallel sum program to sum a series of scalars
 */

//	compile as: gcc sum_scalar.c -lpthread -o sum_scalar
//
//	run as: ./sum_scalar <n> <num_threads>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <limits.h>

#define MAXTHREADS 	8

void *sum(void *p);

// global shared variables
unsigned long int psum[MAXTHREADS];  // partial sum computed by each thread
unsigned long int sumtotal = 0;
unsigned long int n;
int numthreads;
pthread_mutex_t mutex;

int main(int argc, char **argv) {
	pthread_t tid[MAXTHREADS];
	int i, myid[MAXTHREADS];
	struct timeval start, end;

	gettimeofday(&start, NULL); /* start timing */

		scanf("%lu %d", &n, &numthreads);
		for (i = 0; i < numthreads; i++) {
			myid[i] = i;
			psum[i] = 0.0;
			pthread_create(&tid[i], NULL, sum, &myid[i]);
		}
		for (i = 0; i < numthreads; i++) {
			pthread_join(tid[i], NULL);
		}
		pthread_mutex_destroy(&mutex);
	gettimeofday(&end, NULL); /* end timing */
	long spent = (end.tv_sec * 1000000 + end.tv_usec)
			- (start.tv_sec * 1000000 + start.tv_usec);

	printf("%lu\n%ld\n", sumtotal, spent);

	return 0;
}
//-----------------------------------------------
void *sum(void *p) {
	int myid = *((int *) p);
	unsigned long int start = (myid * (unsigned long int) n) / numthreads;
	unsigned long int end = ((myid + 1) * (unsigned long int) n) / numthreads;
	unsigned long int i;

	for (i = start; i < end; i++) {
		psum[myid] += 2;
	}

	pthread_mutex_lock(&mutex);
	sumtotal += psum[myid];
	pthread_mutex_unlock(&mutex);

	return NULL ;
}
