#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Vector length
#define THREADS_PER_BLOCK 256  // Constant threads per block

// Kernel for vector addition
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

// Host function to initialize and launch kernels
int main() {
    int *h_a, *h_b, *h_c;  // Host vectors
    int *d_a, *d_b, *d_c;  // Device vectors
    size_t size = N * sizeof(int);
    
    // Allocate memory on host
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    
    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate memory on device
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Calculate number of blocks required
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel with variable number of blocks and constant threads per block
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Print some results
    printf("Sample results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Free memory
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
