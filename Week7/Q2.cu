#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STR 1024

__global__ void buildRS(const char *S, char *RS, int n) {
    int totalLen = n * (n + 1) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalLen) return;

    int segment = 0; 
    int segmentStart = 0; 
    while (segment < n) {
        int segLen = n - segment;  
        if (idx < segmentStart + segLen) {
            int offset = idx - segmentStart;
            RS[idx] = S[offset];
            return;
        }
        segmentStart += segLen;
        segment++;
    }
}

int main(void) {
    char h_S[MAX_STR];
    
    printf("Enter a string S:\n");
    fgets(h_S, MAX_STR, stdin);
    h_S[strcspn(h_S, "\n")] = '\0';
    
    int n = strlen(h_S);
    if (n == 0) {
        printf("Empty string entered.\n");
        return 0;
    }
    
    int totalLen = n * (n + 1) / 2;
    
    char *h_RS = (char*)malloc((totalLen + 1) * sizeof(char));

    
    char *d_S, *d_RS;
    
    cudaMalloc((void**)&d_S, n * sizeof(char));
    cudaMalloc((void**)&d_RS, totalLen * sizeof(char));
    
    cudaMemcpy(d_S, h_S, n * sizeof(char), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocks = (totalLen + threadsPerBlock - 1) / threadsPerBlock;
    buildRS<<<blocks, threadsPerBlock>>>(d_S, d_RS, n);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_RS, d_RS, totalLen * sizeof(char), cudaMemcpyDeviceToHost);
    h_RS[totalLen] = '\0';  
    printf("Input S : %s\n", h_S);
    printf("Output RS: %s \n", h_RS);
    
    cudaFree(d_S);
    cudaFree(d_RS);
    free(h_RS);
    
    return 0;
}
