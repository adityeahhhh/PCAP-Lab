#include <stdio.h>
#include <cuda_runtime.h>

#define N 16
#define FILTER_SIZE 5

__constant__ int d_filter[FILTER_SIZE];

__global__ void conv1D(const int *input, int *output, int inputSize, int filterSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < inputSize) {
        int sum = 0;
        for (int j = 0; j < filterSize; j++) {
            int idx = i - filterSize / 2 + j;
            if (idx >= 0 && idx < inputSize) {
                sum += input[idx] * d_filter[j];
            }
        }
        output[i] = sum;
    }
}

int main() {
    int h_input[N], h_output[N], h_filter[FILTER_SIZE] = {1, 2, 3, 2, 1};
    int *d_input, *d_output;


    printf("Input Signal:\n");
    for (int i = 0; i < N; i++) {
        h_input[i] = i + 1;
        printf("%d ", h_input[i]);
    }
    printf("\n");


    cudaMemcpyToSymbol(d_filter, h_filter, sizeof(int) * FILTER_SIZE);


    cudaMalloc((void**)&d_input, sizeof(int) * N);
    cudaMalloc((void**)&d_output, sizeof(int) * N);


    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);


    int blockSize = 8;
    int numBlocks = (N + blockSize - 1) / blockSize;
    conv1D<<<numBlocks, blockSize>>>(d_input, d_output, N, FILTER_SIZE);


    cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);


    printf("Convolved Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
