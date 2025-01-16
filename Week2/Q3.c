#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *array = NULL;
    int element;
    MPI_Status status;

    if (rank == 0) {
        array = (int *)malloc(size * sizeof(int));
        printf("Enter %d elements: ", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &array[i]);
        }

        int buffer_size = MPI_BSEND_OVERHEAD + size * sizeof(int);
        void *buffer = malloc(buffer_size);
        MPI_Buffer_attach(buffer, buffer_size);

        for (int i = 1; i < size; i++) {
            MPI_Bsend(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        element = array[0];
        printf("Process 0: Received element %d\n", element);
        printf("Process 0: Square = %d\n", element * element);

        MPI_Buffer_detach(&buffer, &buffer_size);
        free(buffer);
    } else {
        MPI_Recv(&element, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: Received element %d\n", rank, element);

        if (rank % 2 == 0) {
            printf("Process %d: Square = %d\n", rank, element * element);
        } else {
            printf("Process %d: Cube = %d\n", rank, element * element * element);
        }
    }

    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();
    return 0;
}
