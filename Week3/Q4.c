#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    char *s1 = NULL, *s2 = NULL;
    int string_length;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the first string (same length for both): ");
        scanf("%ms", &s1);
        printf("Enter the second string: ");
        scanf("%ms", &s2);
        
        string_length = strlen(s1);
        
        if (string_length % size != 0) {
            printf("Error: String length must be divisible by the number of processes.\n");
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int segment_length = string_length / size;
    char *local_s1 = (char*)malloc((segment_length + 1) * sizeof(char));
    char *local_s2 = (char*)malloc((segment_length + 1) * sizeof(char));
    char *local_result = (char*)malloc((segment_length * 2 + 1) * sizeof(char));

    MPI_Scatter(s1, segment_length, MPI_CHAR, local_s1, segment_length, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, segment_length, MPI_CHAR, local_s2, segment_length, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < segment_length; i++) {
        local_result[2 * i] = local_s1[i];
        local_result[2 * i + 1] = local_s2[i];
    }
    local_result[2 * segment_length] = '\0';
    char *result = NULL;
    if (rank == 0) {
        result = (char*)malloc((string_length * 2 + 1) * sizeof(char));
    }
    MPI_Gather(local_result, segment_length * 2, MPI_CHAR, result, segment_length * 2, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Resultant string: %s\n", result);
        free(s1);
        free(s2);
        free(result);
    }

    free(local_s1);
    free(local_s2);
    free(local_result);

    MPI_Finalize();
    return 0;
}
