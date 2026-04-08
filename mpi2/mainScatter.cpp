#include<iostream>
#include "mpi.h"
#include<vector>
int main(int argc, char* argv[]){
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    std::vector<double> send(size);
    std::vector<double> recv;
    if(0==rank){
        recv.resize(size);
    }
    double data=1.0+rank*rank;
    MPI_Gather(&data,1,MPI_DOUBLE, &recv[0], 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if (0==rank){
        for(int i=0;i<size;++i){
            std::cout<<"data["<<i<<"] "<<recv[i]<<std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}