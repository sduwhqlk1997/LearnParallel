#include <stdio.h>
#include <omp.h>

int main() {
    // 设为4线程
    omp_set_num_threads(4);
    // 并行区域
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("Thread %d/%d: Hello OpenMP\n", tid, nthreads);
    }
    return 0;
}