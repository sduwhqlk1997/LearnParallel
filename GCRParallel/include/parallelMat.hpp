#ifndef PARALLEL_MAT_HPP
#define PARALLEL_MAT_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <mpi.h>
#include <iostream>
using SparseMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
class ParallelMatrix
{
public:
    // SparseMat A;
    int rank, size;
    int global_rows, global_cols;
    std::vector<int> all_rows;
    std::vector<int> start_rows;

    SparseMat localA;
    int local_rows, local_cols;
    ParallelMatrix(){
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    ~ParallelMatrix() = default;
    void distribute(SparseMat A,int rows,int cols)
    {
        global_rows = rows;
        global_cols = cols;
        all_rows.resize(size);
        start_rows.resize(size);
        if (0 == rank)
        {
            int base = global_rows / size;
            int rem = global_rows % size;
            start_rows[0] = 0;
            for (int i = 0; i < size; i++)
            {
                all_rows[i] = base + (i < rem ? 1 : 0);
                if (i > 0)
                    start_rows[i] = start_rows[i - 1] + all_rows[i - 1];
            }
        }
        MPI_Bcast(all_rows.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(start_rows.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

        int my_rows = all_rows[rank];
        int my_start = start_rows[rank];
        // 分发稀疏矩阵数据
        std::vector<double> send_vals, recv_vals;
        std::vector<int> send_inners, recv_inners;
        std::vector<int> send_outers, recv_outers;

        std::vector<int> nnz_list(size, 0);
        std::vector<int> displs_list(size, 0);
        std::vector<int> outer_counts(size, 0);
        std::vector<int> outer_displs(size, 0);

        if (0 == rank)
        {
            const double *val = A.valuePtr();   // 值
            const int *inn = A.innerIndexPtr(); // 列
            const int *out = A.outerIndexPtr(); // 行

            int disp = 0;
            int outer_disp = 0;
            for (int i = 0; i < size; i++)
            {
                int s = start_rows[i];
                int e = s + all_rows[i];
                int nnz = out[e] - out[s]; // out[e]：前e行所有非零元素个数

                nnz_list[i] = nnz;
                displs_list[i] = disp;
                disp += nnz;

                outer_counts[i] = all_rows[i] + 1;
                outer_displs[i] = outer_disp;
                outer_disp += outer_counts[i];

                for (int k = out[s]; k < out[e]; k++)
                {
                    send_vals.push_back(val[k]);
                    send_inners.push_back(inn[k]);
                }

                send_outers.push_back(0);
                int acc = 0;
                for (int r = s; r < e; r++)
                {
                    acc += out[r + 1] - out[r];
                    send_outers.push_back(acc);
                }
            }
        }
        int my_nnz = 0;
        MPI_Scatter(nnz_list.data(), 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        recv_vals.resize(my_nnz);
        recv_inners.resize(my_nnz);
        recv_outers.resize(my_rows + 1);

        MPI_Scatterv(send_vals.data(),
                     nnz_list.data(),
                     displs_list.data(),
                     MPI_DOUBLE,
                     recv_vals.data(),
                     my_nnz, MPI_DOUBLE, 
                     0, MPI_COMM_WORLD);

        MPI_Scatterv(send_inners.data(),
                     nnz_list.data(),
                     displs_list.data(),
                     MPI_INT,
                     recv_inners.data(),
                     my_nnz, MPI_INT, 
                     0, MPI_COMM_WORLD);

        MPI_Scatterv(send_outers.data(),
                     outer_counts.data(),
                     outer_displs.data(),
                     MPI_INT,
                     recv_outers.data(),
                     my_rows + 1, MPI_INT, 
                     0, MPI_COMM_WORLD);

        localA.resize(my_rows, global_cols);
        localA.resizeNonZeros(my_nnz);

        memcpy(localA.valuePtr(), recv_vals.data(), my_nnz * sizeof(double));
        memcpy(localA.innerIndexPtr(), recv_inners.data(), my_nnz * sizeof(int));
        memcpy(localA.outerIndexPtr(), recv_outers.data(), (my_rows + 1) * sizeof(int));
        localA.makeCompressed();
        local_rows=my_rows;
        local_cols=global_cols;
    }
    void printLocalMatrix()
    {
        for (int r = 0; r < size; r++)
        {
            if (rank == r)
            {
                std::cout << "rank " << rank <<" has "<< local_rows <<" rows and "<< local_cols <<" cols"<< std::endl;
                for (int k = 0; k < localA.outerSize(); ++k)
                {
                    std::cout << "row " << k << " :" << std::endl;
                    for (SparseMat::InnerIterator it(localA, k); it; ++it)
                    {
                        std::cout << " " << it.col() << ": " << it.value() << "\t";
                    }
                    std::cout << std::endl;
                }
            }
        }
    }
    void multiply(const Eigen::VectorXd &x, Eigen::VectorXd &result)
    {
        Eigen::VectorXd local_result = localA * x;
        result.resize(global_rows);
        std::vector<int> recv_counts(size);
        std::vector<int> displs(size);
        if(0==rank){
            int offset=0;
            for (int i = 0; i < size;i++){
                recv_counts[i]=all_rows[i];
                displs[i]=offset;
                offset+=all_rows[i];
            }
        }
        MPI_Bcast(recv_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Allgatherv(local_result.data(),
                       local_result.size(),
                       MPI_DOUBLE,
                       result.data(),
                       recv_counts.data(),
                       displs.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }
};

#endif // PARALLEL_MAT_HPP