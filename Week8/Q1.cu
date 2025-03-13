#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void rowWiseAdd(int *A, int *B, int *C, int height, int width) {
    int rid = threadIdx.x;
    for (int cid = 0; cid < width; cid++) {
        C[rid * width + cid] = A[rid * width + cid] + B[rid * width + cid];
    }
}

__global__ void columnWiseAdd(int *A, int *B, int *C, int height, int width) {
    int cid = threadIdx.x;
    for (int rid = 0; rid < height; rid++) {
        C[rid * width + cid] = A[rid * width + cid] + B[rid * width + cid];
    }
}

__global__ void elementWiseAdd(int *A, int *B, int *C, int height, int width) {
    int rid = threadIdx.y;
    int cid = threadIdx.x;
    C[rid * width + cid] = A[rid * width + cid] + B[rid * width + cid];
}

int main() {
    int height, width;
    printf("Enter dimensions: ");
    scanf("%d %d", &height, &width);

    int *A = (int *)malloc(height * width * sizeof(int));
    int *B = (int *)malloc(height * width * sizeof(int));
    int *C = (int *)malloc(height * width * sizeof(int));

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < height * width; i++) {
        scanf("%d", &A[i]);
    }

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < height * width; i++) {
        scanf("%d", &B[i]);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, height * width * sizeof(int));
    cudaMalloc(&d_B, height * width * sizeof(int));
    cudaMalloc(&d_C, height * width * sizeof(int));

    cudaMemcpy(d_A, A, height * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, height * width * sizeof(int), cudaMemcpyHostToDevice);

    rowWiseAdd<<<1, height>>>(d_A, d_B, d_C, height, width);
    cudaMemcpy(C, d_C, height * width * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Row-wise Computation:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", C[i * width + j]);
        }
        printf("\n");
    }

    columnWiseAdd<<<1, width>>>(d_A, d_B, d_C, height, width);
    cudaMemcpy(C, d_C, height * width * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Computation:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", C[i * width + j]);
        }
        printf("\n");
    }

    dim3 threadsPerBlock(width, height);
    elementWiseAdd<<<1, threadsPerBlock>>>(d_A, d_B, d_C, height, width);
    cudaMemcpy(C, d_C, height * width * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nElement-wise Computation:\n");
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", C[i * width + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
