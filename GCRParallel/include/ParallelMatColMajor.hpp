#ifndef PARALLEL_MAT_COL_MAJOR_HPP
#define PARALLEL_MAT_COL_MAJOR_HPP
#include <Eigen/Sparse>
#include <vector>
#include <mpi.h>
#include <iostream>
using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using VecInt = std::vector<int>;
using VecDouble = std::vector<double>;
class ParallelMatColMajor
{
public:
    // 全局变量
    //  SparseMat A;
    int rank, size;
    int global_rows, global_cols;
    VecInt all_cols, start_cols;
    // 分布式变量
    SparseMat localA;
    int local_cols, local_rows;

    ParallelMatColMajor()
    {
        // MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
    void distribute(SparseMat A, int rows, int cols)
    {
        global_rows = rows;
        global_cols = cols;
        all_cols.resize(size);
        start_cols.resize(size);
        if (0 == rank)
        {
            int base = global_cols / size;
            int rem = global_cols % size;
            start_cols[0] = 0;
            for (int i = 0; i < size; i++)
            {
                all_cols[i] = base + (i < rem ? 1 : 0);
                if (i > 0)
                    start_cols[i] = start_cols[i - 1] + all_cols[i - 1];
            }
        }
        MPI_Bcast(all_cols.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(start_cols.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int my_cols = all_cols[rank];
        int my_start = start_cols[rank];

        VecDouble send_vals, recv_vals;
        VecInt send_inners, recv_inners;
        VecInt send_outers, recv_outers;

        VecInt nnz_list(size, 0);
        VecInt displs_list(size, 0);
        VecInt outer_counts(size, 0);
        VecInt outer_displs(size, 0);

        if (0 == rank)
        {
            const double *val = A.valuePtr();
            const int *inn = A.innerIndexPtr();
            const int *out = A.outerIndexPtr();

            int disp = 0;
            int outer_disp = 0;
            for (int i = 0; i < size; i++)
            {
                int s = start_cols[i];
                int e = s + all_cols[i];
                int nnz = out[e] - out[s];

                nnz_list[i] = nnz;
                displs_list[i] = disp;
                disp += nnz;

                outer_counts[i] = all_cols[i] + 1;
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
        recv_outers.resize(my_cols + 1);

        MPI_Scatterv(send_vals.data(),
                     nnz_list.data(), displs_list.data(),
                     MPI_DOUBLE,
                     recv_vals.data(), my_nnz,
                     MPI_DOUBLE,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(send_inners.data(),
                     nnz_list.data(), displs_list.data(),
                     MPI_INT,
                     recv_inners.data(), my_nnz,
                     MPI_INT,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(send_outers.data(),
                     outer_counts.data(), outer_displs.data(),
                     MPI_INT,
                     recv_outers.data(), my_cols + 1,
                     MPI_INT,
                     0, MPI_COMM_WORLD);

        localA.resize(global_rows, my_cols);
        localA.resizeNonZeros(my_nnz);

        memcpy(localA.valuePtr(), recv_vals.data(), my_nnz * sizeof(double));
        memcpy(localA.innerIndexPtr(), recv_inners.data(), my_nnz * sizeof(int));
        memcpy(localA.outerIndexPtr(), recv_outers.data(), (my_cols + 1) * sizeof(int));
        localA.makeCompressed();
        local_cols = my_cols;
        local_rows = global_rows;
    }
    void printLocalMatrix()
    {
        for (int r = 0; r < size; r++)
        {
            if (rank == r)
            {
                std::cout << "rank " << rank << " has " << local_rows << " rows and " << local_cols << " cols" << std::endl;
                for (int k = 0; k < localA.outerSize(); ++k)
                {
                    std::cout << "col " << k << " :" << std::endl;
                    for (SparseMat::InnerIterator it(localA, k); it; ++it)
                    {
                        std::cout << " " << it.row() << ": " << it.value() << "\t";
                    }
                    std::cout << std::endl;
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    ~ParallelMatColMajor() = default;
};
#endif // PARALLEL_MAT_COL_MAJOR_HPP