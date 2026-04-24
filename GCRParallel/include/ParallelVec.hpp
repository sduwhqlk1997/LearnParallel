#ifndef PARALLEL_VEC
#define PARALLEL_VEC
#include<Eigen/Dense>
#include<vector>
#include<mpi.h>
#include<iostream>
using EigenVec = Eigen::VectorXd;
using VecInt = std::vector<int>;
using VecDouble = std::vector<double>;
class ParallelVec{
    public:
        int rank, size;
        int global_size;
        VecInt numValue, startIdx;

        // 分布式变量
        EigenVec localV;
        int localSize;

        ParallelVec(){
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
        }
        ParallelVec(int Vsize):global_size(Vsize){
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
        }
        void distribute(VecDouble V){
            global_size = V.size();
            MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            numValue.resize(size);
            startIdx.resize(size);
            if (0 == rank)
            {
                int base = global_size / size;
                int rem = global_size % size;
                startIdx[0] = 0;
                // VecInt displist(size,0);
                for (int i = 0; i < size; i++)
                {
                    numValue[i] = base + (i < rem ? 1 : 0);
                    if(i>0){
                        startIdx[i] = startIdx[i - 1] + numValue[i - 1];
                    }
                        
                }
            }
            MPI_Bcast(numValue.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(startIdx.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
            localSize = numValue[rank];
            // MPI_Scatter(numValue.data(), 1, MPI_INT, &localSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
            VecDouble recvValue(localSize);
            MPI_Scatterv(V.data(), numValue.data(), 
                        startIdx.data(), MPI_DOUBLE, 
                        recvValue.data(), localSize, 
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            localV = Eigen::Map<EigenVec>(recvValue.data(), recvValue.size());
        }
        void printLocalVec(){
            for (int r = 0; r < size;r++){
                if(rank==r){
                    std::cout << "rank " << rank << " has " << localSize << " values " << std::endl;
                    std::cout << localV << std::endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
        ~ParallelVec() = default;
};
#endif