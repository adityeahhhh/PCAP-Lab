#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void odd_even_sort_kernel(int* arr, int n, bool is_odd_phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < n - 1) {
        if (is_odd_phase) {

            if (idx % 2 == 1 && idx + 1 < n && arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        } else {

            if (idx % 2 == 0 && idx + 1 < n && arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
    }
}


void odd_even_sort_cpu(int* arr, int n) {
    bool sorted = false;
    while (!sorted) {
        sorted = true;

        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                sorted = false;
            }
        }

        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                sorted = false;
            }
        }
    }
}


void odd_even_sort_cuda(int* arr, int n) {
    int* d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));  // Allocate device memory
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);  // Copy data to device

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;  // Number of blocks to launch

    bool is_odd_phase = true;
    for (int i = 0; i < n; i++) {

        odd_even_sort_kernel<<<num_blocks, BLOCK_SIZE>>>(d_arr, n, is_odd_phase);
        cudaDeviceSynchronize();

        is_odd_phase = !is_odd_phase;
    }

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);  // Copy the sorted array back to host

    cudaFree(d_arr);  // Free device memory
}

int main() {
    int n = 10;
    int arr[10] = {64, 25, 12, 22, 11, 90, 55, 33, 20, 42};

    printf("Original Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");


    odd_even_sort_cuda(arr, n);

    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
