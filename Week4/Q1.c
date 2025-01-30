#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int size, rank;
    int i;
    int N, fact = 1, factsum;
    int error_code;
    char error_string[MPI_MAX_ERROR_STRING];
    int length;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (i = 1; i <= rank + 1; i++) {
        fact *= i;
    }

    
    error_code = MPI_Scan(&fact, &factsum,1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (error_code != MPI_SUCCESS) {
        MPI_Error_string(error_code, error_string, &length);
        printf("MPI_Scan failed: %s\n", error_string);
        MPI_Finalize();
        return -1;
    }

    
    printf("Factorial Sum for N=%d: %d\n",rank+1, factsum);
    MPI_Finalize();

    return 0;
}
