#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Vector length

// Kernel for approach (a): Single block, N threads
__global__ void vectorAddBlockSizeN(int *a, int *b, int *c, int n) {
    int tid = threadIdx.x;  // Only thread index matters
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}

// Kernel for approach (b): N threads in multiple blocks
__global__ void vectorAddNThreads(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        c[tid] = a[tid] + b[tid];
}


int main() {
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t size = N * sizeof(int);
    

    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    h_c = (int*)malloc(size);
    

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    

    
    // Approach (a): Single block, N threads
    //vectorAddBlockSizeN<<<1, N>>>(d_a, d_b, d_c, N);
    
    // Approach (b): N threads, multiple blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddNThreads<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    

    printf("Sample results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    

    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return 0;
}
