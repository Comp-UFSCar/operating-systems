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


unsigned long int sumtotal = 0;
unsigned long int n;
int numthreads;
pthread_mutex_t mutex;


int main(int argc, char **argv) {
	int i;
	pthread_t workers[MAXTHREADS];
	struct timeval start, end;

	gettimeofday(&start, NULL);

	scanf("%lu %d", &n, &numthreads);
	for (i = 0; i < numthreads; i++) pthread_create(&workers[i], NULL, sum, i);
	for (i = 0; i < numthreads; i++) pthread_join(workers[i], NULL);

	pthread_mutex_destroy(&mutex);
	gettimeofday(&end, NULL);
	long spent = (end.tv_sec * 1000000 + end.tv_usec) -
				 (start.tv_sec * 1000000 + start.tv_usec);

	printf("%lu\n%ld\n", sumtotal, spent);
	return 0;
}


void *sum(void *p) {
	int myid = (int) p;
	unsigned long int start = (myid * (unsigned long int) n) / numthreads;
	unsigned long int end = ((myid + 1) * (unsigned long int) n) / numthreads;
	unsigned long int i;

	unsigned long int my_sum = 0;

	for (i = start; i < end; i++) my_sum += 2;

	pthread_mutex_lock(&mutex);
	sumtotal += my_sum;
	pthread_mutex_unlock(&mutex);

	return NULL;
}
