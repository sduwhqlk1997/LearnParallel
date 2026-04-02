/*learn the reduction operator*/
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
    double pi=0.0;
    omp_set_num_threads(partitions);
    #pragma omp parallel for reduction(+:pi) //归约;使用“omp parallel for”会自动对循环进行分块
    for(int i=0; i<intervals;i++){
        pi+=(h*4.0) / (1+i*h*i*h);
    }
    auto end{steady_clock::now()};
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"PI is "<<pi<<"\tTime is: "<<std::fixed<<std::setprecision(15)<<time/1000<<std::endl;
    return 0;
}