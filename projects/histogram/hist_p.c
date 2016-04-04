#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


// `hist_p.min_in`
//
// Compute minimum value in a vector.
// 
double
min_in(double *vector, int length)
{
  int i;
  double value = vector[0];

  for (i = 1; i < length; i++)
    if (vector[i] < value) value = vector[i];

  return value;
}


// `hist_p.max_in`
// Compute maximum value in a vector.
double
max_in(double *vector, int length)
{
  int i;
  double value = vector[0];

  for (i = 1; i < length; i++)
    if (vector[i] > value) value = vector[i];

  return value;
}


// `hist_p.count`
//
// Count elements of an array that are in between a range.
//
void
count(int *vector, double min, double max, int n_bins, double h, double *val, int nval)
{
  int i, j, count;
  double min_t, max_t;

  for(j = 0; j < n_bins; j++)
  {
    count = 0;
    min_t = min + j*h;
		max_t = min + (j+1)*h;
		
		for(i = 0; i < nval; i++)
		{
			if(val[i] <= max_t && val[i] > min_t) {
				count++;
			}
		}

		vet[j] = count;
	}

	return vet;
}

int main(int argc, char * argv[]) {
	double h, *val, max, min;
	int n, nval, i, *vet, size;
	long unsigned int duracao;
	struct timeval start, end;

	scanf("%d",&size);

	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */
	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	gettimeofday(&start, NULL);

	/* chama a funcao */
	count(min, max, vet, n, h, val, nval);

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);	
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */	
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");

	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	free(vet);
	free(val);

	return 0;
}
