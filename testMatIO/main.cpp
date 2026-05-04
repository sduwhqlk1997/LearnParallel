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

    /*
    // ===================== 3. 提取稀疏矩阵基础信息 =====================
    int rows = matVar->dims[0];    // 矩阵行数
    int cols = matVar->dims[1];    // 矩阵列数
    int nnz  = matVar->nbytes / sizeof(double); // 非零元素个数（实部+虚部）

    std::cout << "矩阵尺寸: " << rows << " x " << cols << std::endl;
    std::cout << "非零元素个数: " << nnz / 2 << std::endl; // 复数：实部+虚部

    // ===================== 4. 提取 行索引、列索引、实部、虚部 =====================
    // matio 内部存储：稀疏矩阵数据存在 data->sparse 结构体中
    mat_sparse_t* sparseData = static_cast<mat_sparse_t*>(matVar->data);
    
    // 索引（MATLAB：1 开头 → C++：0 开头，后面要减 1）
    int* ir = sparseData->ir;     // 非零元素 行索引
    int* jc = sparseData->jc;     // 列指针（CSC 格式）
    // 数值
    double* pr = sparseData->data; // 实部
    double* pi = pr + nnz / 2;     // 虚部（紧跟在实部后面）

    // ===================== 5. 存入 Eigen 稀疏矩阵 =====================
    SpMatrixXd eigenMat(rows, cols);
    // 预分配非零元素（提升效率）
    eigenMat.reserve(nnz / 2);

    // CSC 格式遍历（Eigen 和 MATLAB 都是 CSC 存储）
    for (int col = 0; col < cols; ++col) {
        for (int idx = jc[col]; idx < jc[col + 1]; ++idx) {
            int row = ir[idx];         // MATLAB 行索引
            std::complex<double> val(pr[idx], pi[idx]); // 复数值
            eigenMat.insert(row, col) = val; // 插入 Eigen
        }
    }

    // 压缩存储（Eigen 推荐操作）
    eigenMat.makeCompressed();

    // ===================== 6. 验证结果 =====================
    std::cout << "\nEigen 稀疏矩阵读取完成！" << std::endl;
    std::cout << "非零元素数量：" << eigenMat.nonZeros() << std::endl;

    // 打印所有非零元素
    for (int k = 0; k < eigenMat.outerSize(); ++k) {
        for (SpMatrixXd::InnerIterator it(eigenMat, k); it; ++it) {
            std::cout << "位置 (" << it.row() << ", " << it.col() << ") = " 
                      << it.value() << std::endl;
        }
    }

    // ===================== 7. 释放资源 =====================
    Mat_VarFree(matVar);
    Mat_Close(mat);

    return EXIT_SUCCESS;
    */
    return 0;
}