#include<iostream>
#include<iomanip>
#include<vector>
#include"omp.h"

int main(){
    const size_t intervals = 11;
    std::vector<double> a;
    std::vector<double> b;
    for(size_t i = 0; i<intervals;i++){
        a.push_back(i);
        b.push_back(2*i+3);
    }
    int n=a.size();
    int nthreads;
    omp_set_num_threads(4); 
    #pragma omp parallel for schedule(dynamic,3)
        for(int i=0;i<n;++i){
            a[i]=a[i]+b[i];
            std::cout<<"thread id: "+std::to_string(omp_get_thread_num())+ " a["+std::to_string(i)+"]"<<std::endl;
        }   
    return 0;
}