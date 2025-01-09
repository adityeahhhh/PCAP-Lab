#include <stdio.h>
#include <mpi.h>
#include <math.h>
int main(int argc, char *argv[]) {
    int rank,a=12,b=16;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0){
        printf("a+b=%d\n",a+b);
    }else if(rank==1){
        printf("a-b=%d\n",a-b);
    }else if(rank==2){
        printf("a*b=%d\n",a*b);
    }else if(rank==3){
        printf("a/b=%d\n",a/b);
    }else if(rank==4){
        printf("a^b=%f\n",pow(a,b));
    }else if(rank==5){
        printf("a mod b=%d\n",a%b);
    }
    

    MPI_Finalize();
    return 0;
}
