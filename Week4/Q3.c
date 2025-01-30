#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int size, rank;
    int i, j;
    int no, N = 4;
    int mat[N][N], result[N][N], temp_result[N];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                scanf("%d", &mat[i][j]);
            }
        }
    }

    MPI_Bcast(mat, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scan(&mat[rank][0], &temp_result[0], N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Gather(&temp_result, N, MPI_INT, result, N, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Output matrix:\n");
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                printf("%d ", result[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
