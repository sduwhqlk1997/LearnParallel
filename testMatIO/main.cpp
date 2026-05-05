#include <iostream>
#include <Eigen/Sparse>
#include <complex>
#include <matio.h>
using SpMatrixXd = Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor>;

int main(int argc, char *argv[])
{
    // 打开文件
    const char *filename = "../K.mat";
    mat_t *mat = Mat_Open(filename, MAT_ACC_RDONLY);
    if (!mat)
    {
        std::cerr << "错误：无法打开 .mat 文件！" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "成功打开 .mat 文件" << std::endl;
    // 读取稀疏矩阵变量
    matvar_t *matVar = Mat_VarRead(mat, "KIDTV");
    if (!matVar)
    {
        std::cerr << "错误：未找到变量KIDTV" << std::endl;
        Mat_Close(mat);
        return EXIT_FAILURE;
    }
    // 校验：必须是 稀疏矩阵 + 复数 + v7.3
    if (matVar->class_type != MAT_C_SPARSE)
    {
        std::cerr << "错误：不是稀疏矩阵！" << std::endl;
        Mat_VarFree(matVar);
        Mat_Close(mat);
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "KIDTV是稀疏矩阵" << std::endl;
    }
    if (!(matVar->isComplex))
    {
        std::cerr << "错误：不是复值矩阵！" << std::endl;
        Mat_VarFree(matVar);
        Mat_Close(mat);
        return EXIT_FAILURE;
    }
    else
    {
        std::cout << "是复值矩阵" << std::endl;
    }

    int rows = matVar->dims[0];
    int cols = matVar->dims[1];
    // int nnz = matVar->nbytes / sizeof(double); // 非零元素个数（实部+虚部）

    std::cout << "矩阵尺寸: " << rows << " x " << cols << std::endl;
    // std::cout << "非零元素个数: " << nnz / 2 << std::endl; // 复数：实部+虚部

    // ===================== 4. 提取 行索引、列索引、实部、虚部 =====================
    mat_sparse_t *sparseData = static_cast<mat_sparse_t *>(matVar->data);

    // 索引（Matlab：1 开头->C++：0开头）
    mat_uint32_t *ir = sparseData->ir;
    mat_uint32_t *jc = sparseData->jc;
    int nnz = jc[cols];

    std::cout << "非零元素个数:" << nnz << std::endl;
    // 5.读取复数数据
    mat_complex_split_t *complexData = static_cast<mat_complex_split_t *>(sparseData->data);

    double *realPart = static_cast<double *>(complexData->Re);
    double *imagPart = static_cast<double *>(complexData->Im);

    // 6.构造Eigen稀疏矩阵
    SpMatrixXd K(rows, cols);

    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    triplets.reserve(nnz);

    // CSC->Eigen
    for (int j = 0; j < cols; ++j)
    {
        for (int idx = jc[j]; idx < jc[j + 1]; ++idx)
        {
            int i = ir[idx];
            std::complex<double> val(realPart[idx], imagPart[idx]);

            triplets.emplace_back(i, j, val);
        }
    }
    K.setFromSortedTriplets(triplets.begin(), triplets.end());
    K.makeCompressed();
    std::cout << "Eigen 稀疏矩阵构造完成" << std::endl;
    std::cout << "Eigen nnz: " << K.nonZeros() << std::endl;
    for (int k = 0; k < std::min(10, (int)triplets.size()); ++k)
    {
        std::cout << "triplet[" << k << "] = ("
                  << triplets[k].row() << ", "
                  << triplets[k].col() << ") = "
                  << triplets[k].value() << std::endl;
    }

    // ===================== 8. 释放资源 =====================
    Mat_VarFree(matVar);
    Mat_Close(mat);
    return 0;
}