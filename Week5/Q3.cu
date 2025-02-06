#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1024  // Array length
#define THREADS_PER_BLOCK 256  // Constant threads per block

// Kernel to compute sine of angles
__global__ void computeSine(float *angles, float *sine_values, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        sine_values[tid] = sinf(angles[tid]);
}

// Host function to initialize and launch kernel
int main() {
    float *h_angles, *h_sine_values;  // Host arrays
    float *d_angles, *d_sine_values;  // Device arrays
    size_t size = N * sizeof(float);
    
    // Allocate memory on host
    h_angles = (float*)malloc(size);
    h_sine_values = (float*)malloc(size);
    
    // Initialize input array with angles in radians
    for (int i = 0; i < N; i++) {
        h_angles[i] = i * (M_PI / 180.0);  // Convert degrees to radians
    }
    
    // Allocate memory on device
    cudaMalloc((void**)&d_angles, size);
    cudaMalloc((void**)&d_sine_values, size);
    
    // Copy data from host to device
    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);
    
    // Calculate number of blocks required
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Launch kernel with variable number of blocks and constant threads per block
    computeSine<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_angles, d_sine_values, N);
    
    // Copy result back to host
    cudaMemcpy(h_sine_values, d_sine_values, size, cudaMemcpyDeviceToHost);
    
    // Print some results
    printf("Sample results:\n");
    for (int i = 100; i < 110; i++) {
        printf("sin(%.6f) = %.6f\n", h_angles[i], h_sine_values[i]);
    }
    
    // Free memory
    free(h_angles); free(h_sine_values);
    cudaFree(d_angles); cudaFree(d_sine_values);
    
    return 0;
}
