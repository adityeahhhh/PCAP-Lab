#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8 

__global__ void inclusive_scan_kernel(int *d_in, int *d_out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        int temp = d_in[idx];
        for (int offset = 1; offset <= blockDim.x; offset *= 2) {
            int t = __shfl_up_sync(0xFFFFFFFF, temp, offset);
            if (threadIdx.x >= offset) {
                temp += t;
            }
            __syncthreads();
        }
        d_out[idx] = temp;
    }
}

void inclusive_scan(int *h_in, int *h_out, int n) {
    int *d_in, *d_out;

    cudaMalloc((void**)&d_in, n * sizeof(int));
    cudaMalloc((void**)&d_out, n * sizeof(int));

    cudaMemcpy(d_in, h_in, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    inclusive_scan_kernel<<<numBlocks, blockSize>>>(d_in, d_out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void print_array(int *array, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

int main() {
    int h_in[N], h_out[N];

    for (int i = 0; i < N; i++) {
        h_in[i] = i + 1;
    }

    inclusive_scan(h_in, h_out, N);

    printf("Input Array:\n");
    print_array(h_in, N);

    printf("\nInclusive Scan Result:\n");
    print_array(h_out, N);

    return 0;
}
