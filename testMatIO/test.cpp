#include <matio.h>
#include <iostream>

int main()
{
    const char *filename = "K.mat"; // 你的 .mat 文件
    const char *varname = "K";         // 稀疏矩阵变量名（MATLAB里保存的名字）

    // 1. 打开 MAT 文件
    mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
    if (!matfp)
    {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return 1;
    }

    // 2. 读取变量
    matvar_t *matvar = Mat_VarRead(matfp, varname);
    if (!matvar)
    {
        std::cerr << "Error: variable " << varname << " not found" << std::endl;
        Mat_Close(matfp);
        return 1;
    }

    // 3. 检查是否是稀疏矩阵
    if (matvar->class_type != MAT_C_SPARSE)
    {
        std::cerr << "Error: variable is not sparse matrix" << std::endl;
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return 1;
    }

    std::cout << "Sparse matrix detected." << std::endl;

    // 4. 获取 sparse 数据结构
    mat_sparse_t *sparse = (mat_sparse_t *)matvar->data;

    size_t nrows = matvar->dims[0];
    size_t ncols = matvar->dims[1];
    size_t nnz = sparse->nzmax;

    std::cout << "Matrix size: " << nrows << " x " << ncols << std::endl;
    std::cout << "Number of nonzeros: " << nnz << std::endl;

    // 5. 输出前几个非零元素（验证数据是否正确）
    mat_uint32_t *ir = sparse->ir;
    mat_uint32_t *jc = sparse->jc;
    double *data = static_cast<double *>(sparse->data);

    for (size_t j = 0; j < ncols; ++j)
    {
        for (mat_uint32_t idx = jc[j]; idx < jc[j + 1]; ++idx)
        {
            mat_uint32_t i = ir[idx];
            double val = data[idx];

            std::cout << "(" << i << ", " << j << ") = " << val << std::endl;
        }
    }
    // 6. 清理
    Mat_VarFree(matvar);
    Mat_Close(matfp);

    std::cout << "Test finished successfully." << std::endl;
    return 0;
}