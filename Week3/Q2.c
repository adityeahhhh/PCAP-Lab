#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int M;
    int N;
    int *data = NULL;
    int *local_data = NULL;
    double local_average, total_average;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N = size;

    if (rank == 0) {
        printf("Enter the value of M: ");
        scanf("%d", &M);
        data = (int*)malloc(N * M * sizeof(int));
        printf("Enter %d elements:\n", N * M);
        for (int i = 0; i < N * M; i++) {
            scanf("%d", &data[i]);
        }
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_data = (int*)malloc(M * sizeof(int));

    MPI_Scatter(data, M, MPI_INT, local_data, M, MPI_INT, 0, MPI_COMM_WORLD);

    double sum = 0.0;
    for (int i = 0; i < M; i++) {
        sum += local_data[i];
    }
    local_average = sum / M;

    double *averages = NULL;
    if (rank == 0) {
        averages = (double*)malloc(N * sizeof(double));
    }
    MPI_Gather(&local_average, 1, MPI_DOUBLE, averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        sum = 0.0;
        for (int i = 0; i < N; i++) {
            sum += averages[i];
        }
        total_average = sum / N;
        printf("Total average: %f\n", total_average);
        free(data);
        free(averages);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
}
