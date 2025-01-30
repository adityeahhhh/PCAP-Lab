#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_LEN 100

int main(int argc, char *argv[]) {
    int rank, size;
    char input_string[MAX_LEN];
    char res[MAX_LEN];
    int N;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a string of N characters (max %d): ", MAX_LEN);
        fgets(input_string, MAX_LEN, stdin);
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char received_char;
    MPI_Scatter(input_string, 1, MPI_CHAR, &received_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);


    for (int i = 0; i <= rank; i++) {
        printf("%c", received_char);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
