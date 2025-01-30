#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int size, rank;
    int i, j;
    int no, N = 3, occurs = 0, totalocc;
    int mat[N][N];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (i = 0; i < N; i++) {
            printf("Enter %dth row: ", i);
            for (j = 0; j < N; j++) {
                scanf("%d", &mat[i][j]);
            }
        }
    }

    MPI_Bcast(mat, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Enter number to search occurrences: ");
        scanf("%d", &no);
    }

    MPI_Bcast(&no, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < N; i++) {
        if (no == mat[rank][i]) {
            occurs += 1;
        }
    }

    MPI_Reduce(&occurs, &totalocc, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total occurrences of %d: %d\n", no, totalocc);
    }

    MPI_Finalize();
    return 0;
}
