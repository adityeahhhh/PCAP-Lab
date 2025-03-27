#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_WIDTH 2
#define TILE_WIDTH 2
#define WIDTH 4

__global__ void MatMulElementThreadShared(int *a, int *b, int *c) {
    __shared__ int MDs[TILE_WIDTH][TILE_WIDTH];
    __shared__ int NDs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int Pvalue = 0;

    for (int m = 0; m < WIDTH / TILE_WIDTH; ++m) {
        MDs[ty][tx] = a[Row * WIDTH + m * TILE_WIDTH + tx];
        NDs[ty][tx] = b[(m * TILE_WIDTH + ty) * WIDTH + Col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += MDs[ty][k] * NDs[k][tx];
        }

        __syncthreads();
    }

    c[Row * WIDTH + Col] = Pvalue;
}

int main() {
    int *matA, *matB, *matProd;
    int *da, *db, *dc;

    matA = (int*)malloc(WIDTH * WIDTH * sizeof(int));
    matB = (int*)malloc(WIDTH * WIDTH * sizeof(int));
    matProd = (int*)malloc(WIDTH * WIDTH * sizeof(int));

    printf("Enter elements of Matrix A:\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        scanf("%d", &matA[i]);
    }

    printf("Enter elements of Matrix B:\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        scanf("%d", &matB[i]);
    }

    cudaMalloc((void**)&da, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&db, WIDTH * WIDTH * sizeof(int));
    cudaMalloc((void**)&dc, WIDTH * WIDTH * sizeof(int));

    cudaMemcpy(da, matA, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, matB, WIDTH * WIDTH * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridConf(WIDTH / BLOCK_WIDTH, WIDTH / BLOCK_WIDTH);
    dim3 blockConf(BLOCK_WIDTH, BLOCK_WIDTH);

    MatMulElementThreadShared<<<gridConf, blockConf>>>(da, db, dc);

    cudaMemcpy(matProd, dc, WIDTH * WIDTH * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Result of Matrix Multiplication:\n");
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        printf("%d ", matProd[i]);
        if ((i + 1) % WIDTH == 0)
            printf("\n");
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(matA);
    free(matB);
    free(matProd);

    return 0;
}
