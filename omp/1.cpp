#include "omp.h"
#include <iostream>
int main()
{
    // omp_set_num_threads(4); // set the number of threads
    #pragma omp parallel
    {
        size_t tid = omp_get_thread_num();
        std::cout << "Hi!" + std::to_string(tid) + "\n";
    }
    std::cout<<omp_get_max_threads()<<std::endl;
    return 0;
}