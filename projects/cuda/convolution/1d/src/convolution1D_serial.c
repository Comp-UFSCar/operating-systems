#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define Mask_Width 101


void Convolution1D(int *N, int *P, int *M, int n) {
    int i, j;

    for(i=0; i < n; i++){
        int Pvalue = 0;
        int N_start_point = i - ((Mask_Width-1)/2);
        for(j = 0; j < Mask_Width; j++){
            if(N_start_point+j >=0 && N_start_point+j < n){
                Pvalue += N[N_start_point+j]*M[j];
            }
        }
        P[i] = Pvalue;
    }
}

int main(int argc, char *argv[]) {

    /* input, output e máscara */
    int *N , *P, *M;
    int n, i;

    /* Tamanho do vetor */
    scanf("%d",&n);

    /* Alocação dos buffers necessários */
    P = (int *)malloc(n*sizeof(int));
    N = (int *)malloc(n*sizeof(int));
    M = (int *)malloc(sizeof(int)*Mask_Width);

    /* entrada dos valores */
    for(i = 0; i < n ; i++)
        scanf("%d",&N[i]);

    for(i = 0; i < Mask_Width; i++) M[i] = i;

    Convolution1D(N, P, M, n);

    for(i = 0; i < n; i++) printf("%d ", P[i]);
    printf("\n");

    free(P);
    free(N);
    free(M);
}
