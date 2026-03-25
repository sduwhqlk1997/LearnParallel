#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__
#include <random>
class Matrix
{
private:
    /* data */
public:
    int num_rows, num_cols;
    double **value;
    Matrix(/* args */);
    Matrix(int n_row, int n_col);
    Matrix(int n_row, int n_col,bool ifzero);
    ~Matrix();
    Matrix operator*(const Matrix& b) const;
    double& operator()(int i,int j);
    const double& operator()(int i, int j) const;
};

Matrix::Matrix() : num_cols(0), num_rows(0) {}
Matrix::Matrix(int n_row, int n_col)
{
    this->num_rows = n_row;
    this->num_cols = n_col;
    this->value = new double *[n_row];
    for (int i = 0; i < n_row; i++)
    {
        this->value[i] = new double[n_col];
        for (int j = 0; j < n_col; j++)
        {
            this->value[i][j] = (double)rand();
        }
    }
}
Matrix::Matrix(int n_row, int n_col,bool ifzero){
    this->num_rows = n_row;
    this->num_cols = n_col;
    this->value = new double *[n_row];
    for (int i = 0; i < n_row; i++)this->value[i] = new double[n_col]();
}
Matrix::~Matrix()
{
}
Matrix Matrix::operator*(const Matrix& b)const{
    Matrix c(this->num_rows,b.num_cols,true);
    for(size_t i=0; i<c.num_rows; ++i){
        for (size_t j=0; j<c.num_cols; ++j){
            for (size_t k=0; k<this->num_cols;++k){
                c(i,j)+= (*this)(i, k)*b(k,j);
            }
        }  
    }
    return c;
}
double& Matrix::operator()(int i,int j){
    return this->value[i][j];
}
const double& Matrix::operator()(int i, int j) const{
    return this->value[i][j];
}
#endif