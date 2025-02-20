#include <stdio.h>
#include <stdlib.h>

void convolve1D(float* N, float* M, float* P, int width, int mask_width) {
    int mask_half = mask_width / 2;


    for (int i = mask_half; i < width - mask_half; i++) {
        float result = 0.0f;

        for (int j = 0; j < mask_width; j++) {
            int N_idx = i + j - mask_half;
            result += N[N_idx] * M[j];
        }

        P[i] = result;
    }
}

int main() {
    int width = 10;
    int mask_width = 3;


    float N[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float M[] = {0.5f, 0.5f, 0.5f};
    float P[width];


    convolve1D(N, M, P, width, mask_width);


    printf("Input Array (N): ");
    for (int i = 0; i < width; i++) {
        printf("%.2f ", N[i]);
    }
    printf("\n");

    printf("Mask Array (M): ");
    for (int i = 0; i < mask_width; i++) {
        printf("%.2f ", M[i]);
    }
    printf("\n");

    printf("Resultant Array (P): ");
    for (int i = 0; i < width; i++) {
        printf("%.2f ", P[i]);
    }
    printf("\n");

    return 0;
}
