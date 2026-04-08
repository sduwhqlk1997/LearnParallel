/*用mpi实现计算pi*/
#include<iostream>
#include "mpi.h"
double numInt(const double a, const double b, const double h){
    double partial_pi=0.0;
    for(double i=a;i<b;i+=h){
        partial_pi+=4*h/(1+i*i);
    }
    return partial_pi;
}

int main(int argc, char* argv[]){
    int rank,size;
    double partial_pi;
    size_t N=1024*1024;
    double h=1/static_cast<double>(N);
    double pi=0;
    MPI_Init(&argc,&argv);
    double start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    double subWidth=1.0/static_cast<double>(size);
    double start=rank*subWidth;
    double end=(rank+1)*subWidth;
    if (rank+1==size){
        end=1;
    }
    partial_pi=numInt(start,end,h);
    MPI_Reduce(&partial_pi,&pi,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double total_time = end_time - start_time;
    if(0==rank){
        std::cout<<"pi is "<<pi<<std::endl;
        printf("=========================================\n");
        printf("The total time is：%.6f s\n", total_time);
        printf("=========================================\n");
    }
    MPI_Finalize();
    return 0;
}