#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>
#include <mpi.h>
void initRandomMatrix(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, int rows, int cols)
{
    A.resize(rows, cols);
    int nnz_per_row = cols < 10 ? cols : 10;                 // 每列非零元素的数量，最多为10
    A.reserve(Eigen::VectorXi::Constant(cols, nnz_per_row)); // 每列预分配nnz_per_col个非零元素
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < nnz_per_row; ++j)
        {
            int col = rand() % cols;
            double value = static_cast<double>(rand()) / RAND_MAX;
            // A.insert(i, col) = value;
            A.coeffRef(i, col) += value;
        }
    }
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_transpose = A.transpose();
    A = (A + A_transpose) / 2.0; // 使矩阵对称
    A.makeCompressed();
}
int main(int argc, char *argv[])
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> A;
    int rank, size;
    int rows = 4, cols = 4;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (0 == rank)
    {
        initRandomMatrix(A, rows, cols);
        std::cout << "Finish matrix initialization " << std::endl;
        std::cout << "A =" << std::endl;
        for (int k = 0; k < A.outerSize(); ++k)
        {
            std::cout << "row " << k << " :" << std::endl;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it)
            {
                std::cout << " " << it.col() << ": " << it.value() << "\t";
            }
            std::cout << std::endl;
        }
    }
    std::vector<int> all_rows(size);
    std::vector<int> start_rows(size);
    if (0 == rank)
    {
        int base = rows / size;
        int rem = rows % size;
        start_rows[0] = 0;
        for (int i = 0; i < size; i++)
        {
            all_rows[i] = base + (i < rem ? 1 : 0);
            if (i > 0)
                start_rows[i] = start_rows[i - 1] + all_rows[i - 1];
        }
    }
    // 广播行数分配信息
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
    // 每个进程接受自己nnz数量
    int my_nnz = 0;
    MPI_Scatter(nnz_list.data(), 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // std::vector<int> nnz_list(size);
    // std::vector<int> outer_counts(size);
    // std::vector<int> displs_list(size);
    // std::vector<int> outer_displs(size);
    // if (rank == 0)
    // {
    //     int disp = 0;
    //     int outer_disp = 0;
    //     for (int i = 0; i < size; i++)
    //     {
    //         int s = start_rows[i];
    //         int e = start_rows[i] + all_rows[i];
    //         nnz_list[i] = A.outerIndexPtr()[e] - A.outerIndexPtr()[s];
    //         displs_list[i] = disp;
    //         disp += nnz_list[i];
    //         outer_counts[i] = all_rows[i] + 1;
    //         outer_displs[i] = outer_disp;
    //         outer_disp += outer_counts[i];
    //     }
    //     MPI_Scatter(nnz_list.data(), 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // }
    // else
    // {
    //     MPI_Scatter(nullptr, 1, MPI_INT, &my_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // }
    // 接受数据
    recv_vals.resize(my_nnz);
    recv_inners.resize(my_nnz);
    recv_outers.resize(my_rows + 1);

    MPI_Scatterv(send_vals.data(), nnz_list.data(), displs_list.data(), MPI_DOUBLE, recv_vals.data(), my_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(send_inners.data(), nnz_list.data(), displs_list.data(), MPI_INT, recv_inners.data(), my_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(send_outers.data(), outer_counts.data(), outer_displs.data(), MPI_INT, recv_outers.data(), my_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 每个进程重建自己的稀疏矩阵
    Eigen::SparseMatrix<double, Eigen::RowMajor> localA(my_rows, cols);
    localA.resizeNonZeros(my_nnz);
    // 拷贝值
    memcpy(localA.valuePtr(), recv_vals.data(), my_nnz * sizeof(double));
    memcpy(localA.innerIndexPtr(), recv_inners.data(), my_nnz * sizeof(int));
    memcpy(localA.outerIndexPtr(), recv_outers.data(), (my_rows + 1) * sizeof(int));

    // 映射行号
    // for (int i = 0; i < my_nnz; i++)
    // {
    //     localA.innerIndexPtr()[i] -= my_start;
    // }
    localA.makeCompressed();

    // 测试输出
    for (int r = 0; r < size; r++)
    {
        if (rank == r)
        {
            std::cout << "rank: " << rank << std::endl;
            for (int k = 0; k < localA.outerSize(); ++k)
            {
                std::cout << "row " << k << " :" << std::endl;
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(localA, k); it; ++it)
                {
                    std::cout << " " << it.col() << ": " << it.value() << "\t";
                }
                std::cout << std::endl;
            }
        }
    }

    // std::cout << "[rank " << rank << "] 行数: " << my_rows
    //           << "  起始行: " << my_start
    //           << "  非零元数: " << my_nnz << std::endl;

    MPI_Finalize();
    return 0;
}