/// Convolution 1D Parallel.
///
/// Implementation of a 1-dimensional convolution in CUDA, with a placeholding
/// mask and shared memory usage.
///
/// Authors:
///     Lucas Oliveira David.
///     Paulo Finardi.
///
/// Note (in Brazilian Portuguese):
/// Como nosso trabalho final e' relacionado `a redes convolucionais,
/// possuindo um operador convolucao implementado em CUDA, ambos os alunos
/// fizeram esta ultima tarefa juntos.
///
/// ___________________________________________________________________________
/// |                            Performance Analysis                         |
/// ___________________________________________________________________________
/// | input   | CPU_Serial  | GPU_NOShared | GPU_Shared | Speedup (CPU/GPUSM) |
/// ___________________________________________________________________________
/// | arq1.in |  0.080165   |  0.110547    | 0.104868   | 0.7644371972384331  |
/// ___________________________________________________________________________
/// | arq2.in |  9.934354   |  8.289333    | 8.235395   | 1.2062996371151598  |
/// ___________________________________________________________________________
/// | arq3.in |  100.419526 |  96.001373   | 91.386422  | 1.0988451435378443  |
/// ___________________________________________________________________________
///
/// License: MIT (c) 2016
///

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include <math.h>
#include<cuda_runtime_api.h>

#define MASK_WIDTH     101
#define OUT_TILE_WIDTH 512


__global__ void _k_conv1d(int *N, int *M, int *out, int n)
{
    __shared__ int mask_s[MASK_WIDTH];
    __shared__ int N_s[OUT_TILE_WIDTH + MASK_WIDTH - 1];

    int i_out    = blockIdx.x*OUT_TILE_WIDTH + threadIdx.x,
        i_in     = i_out - (MASK_WIDTH -1) / 2,
        i_shared = threadIdx.x;

    while (i_shared < OUT_TILE_WIDTH + MASK_WIDTH -1)
    {
        // Loads N into the shared memory.
        if (i_in > -1 && i_in < n)
            N_s[i_shared] = N[i_in];
        else
            N_s[i_shared] = 0;

        i_in += OUT_TILE_WIDTH;
        i_shared += OUT_TILE_WIDTH;
    }

    i_shared = threadIdx.x;
    while (i_shared < MASK_WIDTH)
    {
        mask_s[i_shared] = M[i_shared];
        i_shared += OUT_TILE_WIDTH;
    }

    __syncthreads();

    if (i_out < n)
    {
        int output = 0;
        for(int j = 0; j < MASK_WIDTH; j++)
            output += N_s[threadIdx.x + j] * mask_s[j];

         out[i_out] = output;
    }
}

int main()
{
    int n;
    scanf("%d",&n);
    int size = n*sizeof(int);

    int *input  = (int *)malloc(size);
    int *mask   = (int *)malloc(sizeof(int)*MASK_WIDTH);
    int *output = (int *)malloc(size);

    int *d_input;
    int *d_mask;
    int *d_output;

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_mask, sizeof(int)*MASK_WIDTH);
    cudaMalloc((void **)&d_output, size);

    for(int i = 0; i < n; i++)
    scanf("%d", &input[i]);

    for(int i = 0; i < MASK_WIDTH; i++)
    mask[i] = i;

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, sizeof(int)*MASK_WIDTH, cudaMemcpyHostToDevice);

    dim3 dimGrid((n-1) / OUT_TILE_WIDTH + 1, 1, 1);
    dim3 dimBlock(OUT_TILE_WIDTH, 1, 1);
    _k_conv1d<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; i++)
    printf("%d ", output[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_mask);
    free(input);
    free(output);
    free(mask);

    return 0;
}
