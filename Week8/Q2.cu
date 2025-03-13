#include <cuda.h>
#include <stdio.h>

__global__ void rowWiseMul(int *A, int *B, int *C, int wa, int wb) {
    int rid = threadIdx.x;
    for (int cid = 0; cid < wb; cid++) {
        int ans = 0;
        for (int k = 0; k < wa; k++) {
            ans += A[rid * wa + k] * B[k * wb + cid];
        }
        C[rid * wb + cid] = ans;
    }
}

__global__ void columnWiseMul(int *A, int *B, int *C, int wa, int wb) {
    int cid = threadIdx.x;
    for (int rid = 0; rid < wa; rid++) {
        int ans = 0;
        for (int k = 0; k < wa; k++) {
            ans += A[rid * wa + k] * B[k * wb + cid];
        }
        C[rid * wb + cid] = ans;
    }
}

__global__ void elementWiseMul(int *A, int *B, int *C, int wa, int wb) {
    int rid = threadIdx.y;
    int cid = threadIdx.x;
    int ans = 0;
    for (int k = 0; k < wa; k++) {
        ans += A[rid * wa + k] * B[k * wb + cid];
    }
    C[rid * wb + cid] = ans;
}

int main() {
    int ha, wa, hb, wb;
    printf("Enter dimensions of A : ");
    scanf("%d %d", &ha, &wa);
    printf("Enter dimensions of B : ");
    scanf("%d %d", &hb, &wb);

    if (wa != hb) {
        printf("Error.\n");
        return -1;
    }

    int *A = (int *)malloc(ha * wa * sizeof(int));
    int *B = (int *)malloc(hb * wb * sizeof(int));
    int *C = (int *)malloc(ha * wb * sizeof(int));

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < ha * wa; i++) {
        scanf("%d", &A[i]);
    }

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < hb * wb; i++) {
        scanf("%d", &B[i]);
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, ha * wa * sizeof(int));
    cudaMalloc(&d_B, hb * wb * sizeof(int));
    cudaMalloc(&d_C, ha * wb * sizeof(int));

    cudaMemcpy(d_A, A, ha * wa * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, hb * wb * sizeof(int), cudaMemcpyHostToDevice);

    rowWiseMul<<<1, ha>>>(d_A, d_B, d_C, wa, wb);
    cudaMemcpy(C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Row-wise Computation:\n");
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", C[i * wb + j]);
        }
        printf("\n");
    }

    columnWiseMul<<<1, wb>>>(d_A, d_B, d_C, wa, wb);
    cudaMemcpy(C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nColumn-wise Computation:\n");
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", C[i * wb + j]);
        }
        printf("\n");
    }

    dim3 threadsPerBlock(wb, ha);
    elementWiseMul<<<1, threadsPerBlock>>>(d_A, d_B, d_C, wa, wb);
    cudaMemcpy(C, d_C, ha * wb * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nElement-wise Computation:\n");
    for (int i = 0; i < ha; i++) {
        for (int j = 0; j < wb; j++) {
            printf("%d ", C[i * wb + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}