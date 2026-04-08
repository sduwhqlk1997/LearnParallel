#include<iostream>
#include "mpi.h"
int main(int argc,char* argv[]){
    int rank, size;
    int* buf;
    buf = new int[size];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(0==rank){
        for(int i=0;i<size;++i){
            buf[i]=1+i*i;
        }
    }
    MPI_Bcast(buf,size,MPI_INT,0,MPI_COMM_WORLD);  //将数组buf发送给所有进程
    for (int i=0;i<size;++i){
        if(i==rank){
            printf("rank %d's buffer is %d\n",i,buf[i]);
        }
    }
    MPI_Finalize();
    return 0;
}