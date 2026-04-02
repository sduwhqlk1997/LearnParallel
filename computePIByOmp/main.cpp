#include "omp.h"
#include<iostream>
#include<future>
#include<iomanip>

int main(int argc, char* argv[]){
    using std::chrono::steady_clock;
    using std::chrono::milliseconds;
    using std::chrono::duration_cast;
    size_t intervals=1024*1024*1024;
    size_t partitions = 2;
    if (argc>=2) partitions=std::stoul(argv[1]);
    double h = 1.0 / intervals;
    auto start{steady_clock::now()};
    double* pi=new double(partitions);
    omp_set_num_threads(partitions);
    #pragma omp parallel
    {
        size_t tid = omp_get_thread_num();
        size_t blocksize = intervals/partitions;
        size_t begin = tid*blocksize;
        size_t end=(tid+1)*blocksize;
        pi[tid]=0.0;
        for(int i=begin; i<end;i++){
            pi[tid]+=(h*4.0) / (1+i*h*i*h);
        }
    }
    double PI=0.0;
    for(int i=0;i<partitions;i++) PI+=pi[i];
    auto end{steady_clock::now()};
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"PI is "<<PI<<"\tTime is: "<<std::fixed<<std::setprecision(15)<<time/1000<<std::endl;
    return 0;
}