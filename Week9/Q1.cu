#include <stdio.h>
#include <cuda.h>

__global__ void spmv_csr_kernel(int *row_ptr, int *col_idx, float *values, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float dot = 0.0f;
        for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++) {
            dot += values[j] * x[col_idx[j]];
        }
        y[row] = dot;
    }
}

int main() {
    // Example sparse matrix in CSR format
    // Matrix:  [10  0   0   0]
    //          [0   20  0   30]
    //          [40  0   50  60]
    int h_row_ptr[] = {0, 1, 3, 6};         // Row pointers
    int h_col_idx[] = {0, 1, 3, 0, 2, 3};  // Column indices
    float h_values[] = {10, 20, 30, 40, 50, 60}; // Non-zero values
    float h_x[] = {1, 2, 3, 4};            // Input vector
    int num_rows = 3;                      // Number of rows in the matrix
    int num_nonzeros = 6;                  // Number of non-zero elements


    float h_y[num_rows];


    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;


    cudaMalloc((void **)&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc((void **)&d_col_idx, num_nonzeros * sizeof(int));
    cudaMalloc((void **)&d_values, num_nonzeros * sizeof(float));
    cudaMalloc((void **)&d_x, num_rows * sizeof(float));
    cudaMalloc((void **)&d_y, num_rows * sizeof(float));


    cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx, num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, num_rows * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    spmv_csr_kernel<<<grid_size, block_size>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);

    cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("%f\n", h_y[i]);
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
