/*learn the reduction operator*/
#include "omp.h"
#include <iostream>
#include <future>
#include <iomanip>

int main(int argc, char *argv[])
{
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    using std::chrono::steady_clock;
    size_t intervals = 1024 * 1024 * 1024;
    size_t partitions = 2;
    if (argc >= 2)
        partitions = std::stoul(argv[1]);
    double h = 1.0 / intervals;
    auto start{steady_clock::now()};
    double pi = 0.0;
    double partial_pi = 0.0;
    omp_set_num_threads(partitions);
    #pragma omp parallel default(none) shared(intervals, pi, h) private(partial_pi) // shared：共享，private：私有
    {
        #pragma omp for reduction(+:pi)
        for (int i = 0; i < intervals; i++)
        {
            // partial_pi += (h * 4.0) / (1 + i * h * i * h);
            pi+= (h * 4.0) / (1 + i * h * i * h);
        }
        // #pragma omp critical
        // {
        //     pi+= partial_pi;
        // }
    }
    auto end{steady_clock::now()};
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "PI is " << pi << "\tTime is: " << std::fixed << std::setprecision(15) << time / 1000 << std::endl;
    return 0;
}