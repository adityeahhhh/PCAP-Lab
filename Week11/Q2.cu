#include <stdio.h>
#include <cuda_runtime.h>

#define MAX_ITEMS 10
#define MAX_FRIENDS 100

typedef struct {
    char name[50];
    float price;
} Item;

__global__ void calculateTotal(float *prices, int *quantities, float *totals, int numItems, int numFriends) {
    int friendIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (friendIndex < numFriends) {
        float total = 0.0f;
        
        for (int i = 0; i < numItems; i++) {
            int qty = quantities[friendIndex * numItems + i];
            if (qty < 0) {
                printf("Error: Negative quantity detected for Friend %d, Item %d\n", friendIndex, i);
            }
            total += prices[i] * qty;
        }
        
        totals[friendIndex] = total;
    }
}

int main() {
    Item items[MAX_ITEMS] = {
        {"Shirt", 20.0},
        {"Jeans", 40.0},
        {"Shoes", 50.0},
        {"Hat", 15.0},
        {"Sunglasses", 30.0},
        {"Bag", 60.0},
        {"Watch", 100.0},
        {"Jacket", 80.0},
        {"Scarf", 25.0},
        {"Wallet", 40.0}
    };

    int numItems = MAX_ITEMS;
    int numFriends;

    printf("Enter number of friends (N): ");
    scanf("%d", &numFriends);

    int quantities[MAX_FRIENDS * MAX_ITEMS];
    float totals[MAX_FRIENDS];

    for (int i = 0; i < numFriends; i++) {
        printf("\nFriend %d, please enter the quantities of the following items:\n", i + 1);
        for (int j = 0; j < numItems; j++) {
            printf("%s (Price: %.2f): ", items[j].name, items[j].price);
            scanf("%d", &quantities[i * numItems + j]);
        }
    }

    float prices[MAX_ITEMS];
    for (int i = 0; i < numItems; i++) {
        prices[i] = items[i].price;
    }

    float *d_prices, *d_totals;
    int *d_quantities;

    cudaMalloc((void**)&d_prices, numItems * sizeof(float));
    cudaMalloc((void**)&d_quantities, numFriends * numItems * sizeof(int));
    cudaMalloc((void**)&d_totals, numFriends * sizeof(float));

    cudaMemcpy(d_prices, prices, numItems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantities, quantities, numFriends * numItems * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize = (numFriends + blockSize - 1) / blockSize;
    calculateTotal<<<gridSize, blockSize>>>(d_prices, d_quantities, d_totals, numItems, numFriends);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
        return -1;
    }

    cudaMemcpy(totals, d_totals, numFriends * sizeof(float), cudaMemcpyDeviceToHost);

    float grandTotal = 0.0f;
    for (int i = 0; i < numFriends; i++) {
        printf("\nFriend %d's total purchase: $%.2f\n", i + 1, totals[i]);
        grandTotal += totals[i];
    }

    printf("\nGrand Total for all friends: $%.2f\n", grandTotal);

    cudaFree(d_prices);
    cudaFree(d_quantities);
    cudaFree(d_totals);

    return 0;
}
