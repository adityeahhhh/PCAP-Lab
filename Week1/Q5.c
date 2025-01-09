#include <stdio.h>
#include <mpi.h>

int factorial(int x){
    if(x==0){
        return 1;
    }
    return x*factorial(x-1);
}

int fibonacci(int x){
    if(x<=1){
        return x;
    }
    return fibonacci(x-1)+fibonacci(x-2);
}

int main(int argc, char *argv[]) {
    int rank;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank%2==1){
        printf("Rank %d: Fibonacci: %d\n",rank,fibonacci(rank));
    }else{
        printf("Rank %d: Factorial: %d\n",rank,factorial(rank));
    }

    MPI_Finalize();
    return 0;
}
