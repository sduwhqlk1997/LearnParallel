#include "parallelMat.hpp"
int main(int argc, char *argv[])
{   int rows = 4, cols = 4;
    ParallelMatrix pm(argc, argv, rows, cols);
    Eigen::VectorXd x(cols);
    if(pm.rank == 0){
        x = Eigen::VectorXd::Random(cols);
    }
    Eigen::VectorXd result;
    pm.distribute();
    if (pm.rank == 0)
    {        
        // std::cout << "This is a new block mat code. Global matrix A:" << std::endl;
        //pm.printGlobalMatrix();
        Eigen::VectorXd global_result = pm.A * x;
        std::cout << "Global result A*x:" << std::endl;
        std::cout << global_result.transpose() << std::endl;
        std::cout<<pm.all_rows[0]<<" "<<pm.all_rows[1]<<" "<<pm.all_rows[2]<<std::endl;    
    }
    // pm.printLocalMatrix();
    MPI_Bcast(x.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    pm.multiply(x, result);
    if (pm.rank == 0)
    {
        std::cout << "Result of parallel multiplication A*x:" << std::endl;
        std::cout << result.transpose() << std::endl;
    }
    return 0;
}   