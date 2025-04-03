#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 4
#define WIDTH 16
#define MASK_WIDTH 5

__global__ void tiledConvolution(float *N, float *M, float *P) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_mask = MASK_WIDTH / 2;

    extern __shared__ float shared_N[];

    if (tid < WIDTH) {
        shared_N[threadIdx.x + half_mask] = N[tid];
        if (threadIdx.x < half_mask) {
            shared_N[threadIdx.x] = (tid - half_mask >= 0) ? N[tid - half_mask] : 0.0f;
            shared_N[threadIdx.x + TILE_WIDTH + half_mask] = (tid + TILE_WIDTH < WIDTH) ? N[tid + TILE_WIDTH] : 0.0f;
        }
    }
    __syncthreads();

    if (tid < WIDTH) {
        float sum = 0.0f;
        for (int i = 0; i < MASK_WIDTH; i++) {
            sum += shared_N[threadIdx.x + i] * M[i];
        }
        P[tid] = sum;
    }
}

int main() {
    float N[WIDTH] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float M[MASK_WIDTH] = {0.1f, 0.2f, 0.5f, 0.2f, 0.1f};  // Averaging filter
    float P[WIDTH];

    float *d_N, *d_M, *d_P;
    cudaMalloc((void**)&d_N, WIDTH * sizeof(float));
    cudaMalloc((void**)&d_M, MASK_WIDTH * sizeof(float));
    cudaMalloc((void**)&d_P, WIDTH * sizeof(float));

    cudaMemcpy(d_N, N, WIDTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = TILE_WIDTH;
    int gridSize = (WIDTH + blockSize - 1) / blockSize;

    int sharedMemSize = (TILE_WIDTH + MASK_WIDTH - 1) * sizeof(float);  // Shared memory size

    tiledConvolution<<<gridSize, blockSize, sharedMemSize>>>(d_N, d_M, d_P);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaMemcpy(P, d_P, WIDTH * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output array P:\n");
    for (int i = 0; i < WIDTH; i++) {
        printf("P[%d] = %.2f\n", i, P[i]);
    }

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
