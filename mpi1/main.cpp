#include <iostream>
#include<unistd.h>
#include "mpi.h"

int main(int argc, char* argv[]){
    MPI_Init(&argc,&argv);
    int tag=0;
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    double recMessage;
    double sendMessage=rank*2.2;
    int left=rank-1;
    if(left==-1) left=size-1;
    int right=rank+1;
    if(right==size) right=0;
    // if(rank%2==0){
    //     MPI_Send(&sendMessage,1,MPI_DOUBLE,right,tag,MPI_COMM_WORLD);
    //     MPI_Recv(&recMessage,1,MPI_DOUBLE,left,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    // }
    // else{
    //     MPI_Recv(&recMessage,1,MPI_DOUBLE,left,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    //     MPI_Send(&sendMessage,1,MPI_DOUBLE,right,tag,MPI_COMM_WORLD);
    // }
    MPI_Sendrecv(&sendMessage,1,MPI_DOUBLE,right,tag,&recMessage,1,MPI_DOUBLE,left,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    std::cout<<"rank: "+std::to_string(rank)+" recMessage: "+std::to_string(recMessage)<<std::endl;
    MPI_Finalize();
    return 0;
}