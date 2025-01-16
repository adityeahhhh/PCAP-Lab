#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size, a, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if (rank == 0) {
        printf("Enter a number to send to slave processes: ");
        scanf("%d", &a);
        
        for (i = 1; i < size; i++) {
            MPI_Send(&a, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        printf("Rank 0: Sent the number '%d' to all slave processes\n", a);

    } else {
        // Receive the number from process 0
        MPI_Recv(&a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Rank %d: Received the number '%d' from master process\n", rank, a);
    }

    MPI_Finalize();
    return 0;
}
