#include <stdio.h>
#include <cuda.h>

#define M 4 // Number of rows
#define N 4 // Number of columns

__global__ void processMatrix(int *A, int *B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // Check if the element is a border element
        if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
            B[row * cols + col] = A[row * cols + col]; // Keep border elements
        } else {
            B[row * cols + col] = ~A[row * cols + col]; // 1's complement for non-border elements
        }
    }
}

int main() {
    int h_A[M][N] = {
        {1, 2, 3, 4},
        {6, 5, 8, 3},
        {2, 4, 10, 1},
        {9, 1, 2, 5}
    };
    int h_B[M][N];

    int *d_A, *d_B;
    size_t size = M * N * sizeof(int);

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    // Copy matrix A from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(2, 2);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    processMatrix<<<gridSize, blockSize>>>(d_A, d_B, M, N);

    // Copy the result matrix B from device to host
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Print the output matrix B
    printf("Output Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
