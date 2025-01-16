#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size, a, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    if(rank==0){
        printf("Enter a number to send to processes: ");
        scanf("%d", &a);
        a+=1;
        MPI_Send(&a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Rank 0: Sent the number '%d' to process 1\n", a);
        MPI_Recv(&a, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, &status);
        printf("Rank 0: Finally received number %d from process %d\n",a,size-1);
    }
    else if (rank+1!=size) {
        MPI_Recv(&a, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
        a+=1;
        MPI_Send(&a, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);

        printf("Rank %d: Sent the number '%d' to process %d\n",rank, a,rank+1);

    } else {
        MPI_Recv(&a, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
        a+=1;   
        MPI_Send(&a, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Rank %d: Sent the number '%d' to process 0\n",rank, a);
    }

    MPI_Finalize();
    return 0;
}
