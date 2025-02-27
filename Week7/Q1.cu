#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SENTENCE 1024
#define MAX_WORD 100

__global__ void countWord(const char* sentence, const char* word, int sentenceLength, int wordLength, unsigned int* d_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > sentenceLength - wordLength)
        return;
    
    bool match = true;
    for (int j = 0; j < wordLength; j++) {
        if (sentence[i + j] != word[j]) {
            match = false;
            break;
        }
    }
    
    if (match && (i == 0 || sentence[i - 1] == ' ' || sentence[i - 1] == '\n') && (i + wordLength == sentenceLength || sentence[i + wordLength] == ' ' || sentence[i + wordLength] == '\n' || sentence[i + wordLength] == '\0')) {
        atomicAdd(d_count, 1);
    }
}

int main() {

    char sentence[MAX_SENTENCE];
    char word[MAX_WORD];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    printf("Enter a sentence:\n");
    fgets(sentence, MAX_SENTENCE, stdin);
    sentence[strcspn(sentence, "\n")] = 0;  // Remove newline character

    printf("Enter the word to search for:\n");
    fgets(word, MAX_WORD, stdin);
    word[strcspn(word, "\n")] = 0;  // Remove newline character

    int sentenceLength = strlen(sentence);
    int wordLength = strlen(word);

    if (sentenceLength < wordLength) {
        printf("The word is longer than the sentence. Occurrences: 0\n");
        return 0;
    }
    
    char *d_sentence, *d_word;
    unsigned int *d_count;
    unsigned int count = 0;
    
    cudaMalloc((void**)&d_sentence, sentenceLength * sizeof(char));
    cudaMalloc((void**)&d_word, wordLength * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    
    cudaMemcpy(d_sentence, sentence, sentenceLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, wordLength * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocks = (sentenceLength - wordLength + 1 + threadsPerBlock - 1) / threadsPerBlock;
    
    countWord<<<blocks, threadsPerBlock>>>(d_sentence, d_word, sentenceLength, wordLength, d_count);

    cudaDeviceSynchronize();
    
    cudaMemcpy(&count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    printf("The word '%s' appears %u times in the sentence.\n", word, count);
    
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    
    return 0;
}
