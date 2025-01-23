#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int is_vowel(char c) {
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
            c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U');
}

int main(int argc, char** argv) {
    int rank, size;
    char *input_string = NULL;
    int string_length, local_non_vowels, total_non_vowels;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a string: ");
        scanf("%ms", &input_string);
        string_length = strlen(input_string);

        if (string_length % size != 0) {
            printf("Error: String length must be divisible by the number of processes.\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char *local_string = (char*)malloc((string_length / size + 1) * sizeof(char));

    MPI_Scatter(input_string, string_length / size, MPI_CHAR, local_string, string_length / size, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_non_vowels = 0;
    for (int i = 0; i < string_length / size; i++) {
        if (!is_vowel(local_string[i])) {
            local_non_vowels++;
        }
    }

    int *non_vowel_counts = NULL;
    if (rank == 0) {
        non_vowel_counts = (int*)malloc(size * sizeof(int));
    }
    MPI_Gather(&local_non_vowels, 1, MPI_INT, non_vowel_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        total_non_vowels = 0;
        for (int i = 0; i < size; i++) {
            printf("Process %d found %d non-vowels.\n", i, non_vowel_counts[i]);
            total_non_vowels += non_vowel_counts[i];
        }
        printf("Total number of non-vowels: %d\n", total_non_vowels);
        free(input_string);
        free(non_vowel_counts);
    }

    free(local_string);
    MPI_Finalize();
    return 0;
}
