#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void _gpu_m_add(int *a, int *b, int *c, int rows, int columns)
{
    int i = TILE_WIDTH * blockIdx.y + threadIdx.y;
    int j = TILE_WIDTH * blockIdx.x + threadIdx.x;

    if (i < rows && j < columns)
        c[i * columns + j] = a[i * columns + j] + b[i * columns + j];
}


int m_add(int *a, int *b, int *c, int rows, int columns)
{
    int *da,
        *db,
        *dc,
        size = rows * columns * sizeof(int);

    cudaMalloc((void **)&da, size);
    cudaMalloc((void **)&db, size);
    cudaMalloc((void **)&dc, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float)columns / TILE_WIDTH),
                 ceil((float)rows / TILE_WIDTH),
                 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    _gpu_m_add<<<dimGrid, dimBlock>>>(da, db, dc, rows, columns);

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);
    cudaFree(da); cudaFree(db); cudaFree(dc);

    return 0;
}


int main()
{
    int *A, *B, *C;
    int i, j;

    //Input
    int linhas, colunas;

    scanf("%d", &linhas);
    scanf("%d", &colunas);

    //Alocando memória na CPU
    A = (int *)malloc(sizeof(int)*linhas*colunas);
    B = (int *)malloc(sizeof(int)*linhas*colunas);
    C = (int *)malloc(sizeof(int)*linhas*colunas);

    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    m_add(A, B, C, linhas, colunas);

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];
        }
    }

    printf("%lli\n", somador);

    free(A);
    free(B);
    free(C);
}
