#include <stdio.h>
#include <mpi.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, x=4;
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("For process with rank %d, pow(x, rank): %f\n", rank, pow(x, rank));

    MPI_Finalize();
    return 0;
}
