#include<iostream>
#include "mpi.h"
#include<vector>
int rank,size;
int main(int argc, char* argv[]){
    int n = 18;
    int localn;
    int maxt = 100;
    double length = 1.0;
    double dt = 0.001;
    double dx = length/n;
    double a=0.25*dt/(dx*dx);
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int left = rank - 1; 
    if(left < 0) left=MPI_PROC_NULL;
    int right = rank + 1;
    if(right>=size) right=MPI_PROC_NULL;

    localn = n/size;
    std::vector<double> T(localn+2);
    std::vector<double> newT(localn+2);

    for(int i=0;i<maxt;++i){
        MPI_Sendrecv(&T[1],1,MPI_DOUBLE,left,0,&T[localn+1],1,MPI_DOUBLE,right,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(&T[localn],1,MPI_DOUBLE,right,1,&T[0],1,MPI_DOUBLE,left,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(rank==0) T[0]=1.0;
        if(rank==size-1) T[localn+1]=0.0;
        for(int j=1;j<=localn;++j){
            newT[j] = T[j]+a*(T[j-1]-2*T[j]+T[j+1]);
        }
        T.swap(newT);
    }
    MPI_Finalize();
    return 0;
}