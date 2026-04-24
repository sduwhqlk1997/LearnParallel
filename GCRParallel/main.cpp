// #include "ParallelMatColMajor.hpp"
#include <stdexcept>
#include <fstream>
#include <unistd.h>
#include "ParallelVec.hpp"
#include "parallelMat.hpp"
void readSparseMatrixFromTxt(const std::string &filePath,
                             SparseMat &sparseMat,
                             int rows = -1,
                             int cols = -1);
bool readVectorFromTxt(const std::string &filePath, std::vector<double> &vec);
bool ParallelGCR(VecDouble &resHis, int &iter, EigenVec &x,
                 const int maxit, const double epcl,
                 const ParallelMatrix A, const ParallelVec b, const int size, const int rank); // 并行GCR算法
int main(int argc, char *argv[])
{
#ifdef DEBUG
    {
        int i = 0;
        while (0 == i)
        {
            sleep(1);
        }
    }
#endif
    SparseMat mat;
    VecDouble vec;
    int rank, size;
    MPI_Init(&argc, &argv);
    ParallelMatrix A;
    ParallelVec f;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (0 == rank) // 从txt文件读取稀疏矩阵并存入mat
    {
        /*读取矩阵*/
        try
        {

            // 方式1：自动推断矩阵行列大小
            readSparseMatrixFromTxt("../A.txt", mat);

            // 方式2：手动指定矩阵行列（推荐，更严谨）
            // readSparseMatrixFromTxt("matrix.txt", mat, 1000, 500);

            std::cout << "读取成功！矩阵大小："
                      << mat.rows() << " × " << mat.cols()
                      << "，非零元素个数：" << mat.nonZeros() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "错误：" << e.what() << std::endl;
            return -1;
        }
        bool success = readVectorFromTxt("../f.txt", vec);
        /*读取向量*/
        if (success)
        {
            std::cout << "读取成功！共读取 " << vec.size() << " 个数字：" << std::endl;
        }
        else
        {
            std::cout << "读取失败！" << std::endl;
        }
        // std::cout << "mat =" << std::endl;
        // for (int k = 0; k < mat.outerSize(); ++k)
        // {
        //     std::cout << "col " << k << " :" << std::endl;
        //     for (SparseMat::InnerIterator it(mat, k); it; ++it)
        //     {
        //         std::cout << " " << it.row() << ": " << it.value() << "\t";
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;
    }
    A.distribute(mat, 3375, 3375);
    f.distribute(vec);

    /*设置初始解*/
    // EigenVec f = Eigen::Map<EigenVec>(vec.data(), vec.size());
    EigenVec x0 = EigenVec::Zero(3375);
    /*测试分片矩阵向量乘法*/
    // EigenVec serrial_result, local_result, global_result;
    // if (0 == rank)
    // {
    //     EigenVec globalvec = Eigen::Map<EigenVec>(vec.data(), vec.size());
    //     serrial_result = mat * globalvec;
    //     global_result.resize(vec.size());
    // }
    // local_result = A.localA * f.localV;
    // MPI_Reduce(local_result.data(), global_result.data(), local_result.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // if (0 == rank)
    // {
    //     EigenVec err = global_result - serrial_result;
    //     double res = err.norm();
    //     res /= serrial_result.norm();
    //     std::cout << "分片矩阵乘向量误差为：" << res << std::endl;
    // }
    /*GCR迭代*/
    VecDouble resHis;
    int maxit = 50;
    int iter;
    double epcl = 1e-4;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    bool flag = ParallelGCR(resHis, iter, x0, maxit, epcl, A, f, size, rank);
    double end = MPI_Wtime();
    if (rank == 0)
    {
        printf("运行时间: %.6f 秒\n", end - start);
    }
    // for (int i = 0; i < resHis.size(); i++)
    // {
    //     std::cout << resHis[i] << std::endl;
    // }
    // f.printLocalVec();
    MPI_Finalize();
    return 0;
}
void readSparseMatrixFromTxt(const std::string &filePath,
                             SparseMat &sparseMat,
                             int rows,
                             int cols)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("无法打开文件：" + filePath);
    }

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> triplets;

    // 支持读取科学计数法浮点数格式的行列索引
    double i, j;
    double val;

    int max_row = 0, max_col = 0;

    // 正常读取你的数据
    while (file >> i >> j >> val)
    {
        // 1起始索引 → 0起始索引
        int eigen_i = static_cast<int>(i) - 1;
        int eigen_j = static_cast<int>(j) - 1;

        triplets.emplace_back(eigen_i, eigen_j, val);

        // 记录最大行列
        if (static_cast<int>(i) > max_row)
            max_row = static_cast<int>(i);
        if (static_cast<int>(j) > max_col)
            max_col = static_cast<int>(j);
    }

    file.close();

    int mat_rows = (rows > 0) ? rows : max_row;
    int mat_cols = (cols > 0) ? cols : max_col;

    if (mat_rows <= 0 || mat_cols <= 0)
    {
        throw std::runtime_error("矩阵尺寸无效，请检查文件是否为空或索引错误");
    }

    // 赋值给 行优先 稀疏矩阵
    sparseMat.resize(mat_rows, mat_cols);
    sparseMat.setFromTriplets(triplets.begin(), triplets.end());
}
bool readVectorFromTxt(const std::string &filePath, std::vector<double> &vec)
{
    // 先清空向量，避免旧数据干扰
    vec.clear();

    // 打开文件（只读模式）
    std::ifstream inFile(filePath);

    // 判断文件是否成功打开
    if (!inFile.is_open())
    {
        std::cerr << "错误：无法打开文件 -> " << filePath << std::endl;
        return false;
    }

    double num;
    // 循环读取所有数字（自动跳过空格、换行、制表符）
    while (inFile >> num)
    {
        vec.push_back(num);
    }

    // 检查是否因读取错误结束循环（非文件结束）
    if (!inFile.eof())
    {
        std::cerr << "错误：读取文件时发生异常！" << std::endl;
        inFile.close();
        return false;
    }

    // 关闭文件
    inFile.close();
    return true;
}
bool ParallelGCR(VecDouble &resHis, int &iter, EigenVec &x,
                 const int maxit, const double epcl,
                 const ParallelMatrix A, const ParallelVec b, const int size, const int rank)
{
    EigenVec r_local(b.localSize);
    EigenVec r_global(b.global_size);
    // std::cout << "local_rows: " << A.local_rows<< ", local_cols: " << A.local_cols << std::endl;
    // std::cout << "rank: " << rank << "localA_size: "<<A.localA.rows()<< ", "<<A.localA.cols()<<" xsize: "<<x.size()<<std::endl;
    r_local = A.localA * x;
    // std::cout << "rank: " << rank << std::endl;
    r_local = b.localV - r_local;
    MPI_Allgatherv(
        r_local.data(), b.localSize, MPI_DOUBLE,
        r_global.data(), b.numValue.data(), b.startIdx.data(),
        MPI_DOUBLE, MPI_COMM_WORLD);
    double res_local = r_local.squaredNorm();
    double res;
    MPI_Allreduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    res = sqrt(res);
    // double res = r_global.norm();
    resHis.push_back(res);
    if (res <= epcl)
        return true;
    std::vector<EigenVec> p;
    EigenVec p_k;
    p.push_back(r_global);
    VecDouble ApAp;
    std::vector<EigenVec> Ap;
    EigenVec Ap_k = A.localA * p[0];
    // std::cout << "rank: " << rank << "localA_size: "<<A.localA.rows()<< ", "<<A.localA.cols()<<" psize: "<<p[0].size()<<std::endl;
    Ap.push_back(Ap_k);
    double ApAp_k = Ap_k.squaredNorm();
    double ApAp_result;
    MPI_Allreduce(&ApAp_k, &ApAp_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ApAp.push_back(ApAp_result);
    double alpha, local_alpha;
    EigenVec local_Ar;
    EigenVec beta, local_beta;
    for (int k = 1; k < maxit; k++)
    {
        local_alpha = r_local.dot(Ap[k - 1]);
        local_alpha = local_alpha / ApAp[k - 1];
        MPI_Allreduce(&local_alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        x += alpha * p[k - 1];
        r_local -= alpha * Ap[k - 1];
        // MPI_Gatherv(r_local.data(), b.localSize, MPI_DOUBLE,
        //             r_global.data(), b.numValue.data(), b.startIdx.data(),
        //             MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Bcast(r_global.data(), b.global_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Allgatherv(
            r_local.data(), b.localSize, MPI_DOUBLE,
            r_global.data(), b.numValue.data(), b.startIdx.data(),
            MPI_DOUBLE, MPI_COMM_WORLD);
        res_local = r_local.squaredNorm();
        MPI_Allreduce(&res_local, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        res = sqrt(res);
        // res = r_global.norm();
        resHis.push_back(res);
        if (res <= epcl)
        {
            iter = k;
            if (0 == rank)
                std::cout << "GCR algorithm converged in " << k << "steps with residual= " << res << "." << std::endl;
            return true;
        }
        local_Ar = A.localA * r_global;
        beta.resize(k);
        local_beta.resize(k);
        for (int i = 0; i < k; i++)
        {
            local_beta[i] = -local_Ar.dot(Ap[i]);
            local_beta[i] /= ApAp[i];
        }
        MPI_Allreduce(local_beta.data(), beta.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        p_k = r_global;
        for (int i = 0; i < k; i++)
        {
            p_k += beta[i] * p[i];
        }
        p.push_back(p_k);
        Ap_k = local_Ar;
        for (int i = 0; i < k; i++)
        {
            Ap_k += beta[i] * Ap[i];
        }
        ApAp_k = Ap_k.squaredNorm();
        MPI_Allreduce(&ApAp_k, &ApAp_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        ApAp.push_back(ApAp_result);
        Ap.push_back(Ap_k);
    }
    return false;
}