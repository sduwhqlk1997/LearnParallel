#include <iostream>
#include "matrix.hpp"
#include "timer.hpp"
#ifndef mulType
#define mulType 0
#endif
using namespace std;
void mul22(const Matrix &A, const Matrix &B, Matrix &C)
{
    for (size_t i = 0; i < A.num_rows; i += 2)
    {
        for (size_t j = 0; j < B.num_cols; j += 2)
        {
            for (size_t k = 0; k < A.num_cols; k += 2)
            {
                C(i, j) += A(i, k) * B(k, j);
                C(i, j) += A(i, k + 1) * B(k + 1, j);
                C(i, j + 1) += A(i, k) * B(k, j + 1);
                C(i, j + 1) += A(i, k + 1) * B(k + 1, j + 1);
                C(i + 1, j) += A(i + 1, k) * B(k, j);
                C(i + 1, j) += A(i + 1, k + 1) * B(k + 1, j);
                C(i + 1, j + 1) += A(i + 1, k) * B(k, j + 1);
                C(i + 1, j + 1) += A(i + 1, k + 1) * B(k + 1, j + 1);
            }
        }
    }
}
int main()
{
    cout << "N\tduration" << endl;

    for (int N = 8; N < 512; N *= 2)
    {
        Matrix A(N, N, true), B(N, N), C(N, N);
        Timer T;
        T.start();
        switch (mulType)
        {
        case 0:
            A = B * C;
            break;
        case 1:
            mul22(B, C, A);
            break;
        default:
            continue;
            break;
        }
        T.end();
        cout << N << "\t" << T.duration() << endl;
    }
    return 0;
}