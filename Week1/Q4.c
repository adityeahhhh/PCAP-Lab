#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;
    char str1[]="HELLO";
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(str1[rank]>92){
        str1[rank]=str1[rank]-32;
    }else{
        str1[rank]=str1[rank]+32;
    }

    printf("%s\n",str1);
    MPI_Finalize();
    return 0;
}