#include <stdio.h>
#include <mpi.h>

int factorial(int a){
	if(a==0){
		return 1;
	}
	return a*factorial(a-1);
}

int main(int argc,char *argv[]){
	int rank,size,N,A[10],B[10],c,d,i,sum=0;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	if(rank==0){
		N=size;
		fprintf(stdout,"Enter %d values:\n",N);
		fflush(stdout);
		for(i=0;i<N;i++){
			scanf("%d",&A[i]);
		}
	}
	MPI_Scatter(A,1,MPI_INT,&c,1,MPI_INT,0,MPI_COMM_WORLD);
	fprintf(stdout,"Process %d has recieved %d\n",rank,c);
	fflush(stdout);
	d=factorial(c);
	MPI_Gather(&d,1,MPI_INT,B,1,MPI_INT,0,MPI_COMM_WORLD);

	if(rank==0){
		fprintf(stdout,"The result gathered in the root \n");
		fflush(stdout);
		for(i=0;i<N;i++){
			sum+=B[i];
		}
		fprintf(stdout,"%d\n",sum);
		fflush(stdout);
	}
	MPI_Finalize();
	return 0;
}