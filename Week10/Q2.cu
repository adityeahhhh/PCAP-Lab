#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4   
#define BLOCK_SIZE 2  

__global__ void matrixMulKernel(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // global col index

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);


    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);


    printf("Enter %d elements for Matrix A (%dx%d):\n", N*N, N, N);
    for (int i = 0; i < N * N; i++) {
        scanf("%d", &h_A[i]);
    }

    printf("Enter %d elements for Matrix B (%dx%d):\n", N*N, N, N);
    for (int i = 0; i < N * N; i++) {
        scanf("%d", &h_B[i]);
    }


    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);


    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    printf("Resultant Matrix C (A x B):\n");
    for (int i = 0; i < N * N; i++) {
        printf("%d ", h_C[i]);
        if ((i + 1) % N == 0) {
            printf("\n");
        }
    }


    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
