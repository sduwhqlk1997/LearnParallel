#include<iostream>
#include "mpi.h"
#include<vector>
int rank,size;
int main(int argc, char* argv[]){
    int n = 18;
    int localn;
    int maxt = 10000;
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
    MPI_Finalize();
    return 0;
}