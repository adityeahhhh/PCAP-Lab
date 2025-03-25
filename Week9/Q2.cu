#include <stdio.h>
#include <cuda.h>

// Kernel to modify the matrix rows
__global__ void modify_matrix(float *matrix, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        for (int col = 0; col < N; col++) {
            float base = matrix[row * N + col];
            matrix[row * N + col] = powf(base, row + 1);
        }
    }
}

int main() {
    int M = 4; // Number of rows
    int N = 5; // Number of columns


    float h_matrix[M * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    };

    printf("Original Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_matrix[i * N + j]);
        }
        printf("\n");
    }


    float *d_matrix;
    cudaMalloc((void **)&d_matrix, M * N * sizeof(float));


    cudaMemcpy(d_matrix, h_matrix, M * N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (M + threads_per_block - 1) / threads_per_block;
    modify_matrix<<<blocks_per_grid, threads_per_block>>>(d_matrix, M, N);

    cudaMemcpy(h_matrix, d_matrix, M * N * sizeof(float), cudaMemcpyDeviceToHost);


    printf("\nModified Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_matrix[i * N + j]);
        }
        printf("\n");
    }


    cudaFree(d_matrix);

    return 0;
}
