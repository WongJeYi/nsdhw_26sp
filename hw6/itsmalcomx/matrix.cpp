#include "matrix.hpp"
#include <algorithm>
#include <stdexcept>
#ifdef USE_CBLAS
#include <cblas.h>
#else
#include <mkl.h>
#endif

Matrix multiply_naive(const Matrix & A, const Matrix & B)
{
    if (A.ncol() != B.nrow()) throw std::invalid_argument("dimension mismatch");
    const size_t M=A.nrow(), K=A.ncol(), N=B.ncol();
    Matrix C(M,N);
    for (size_t i=0;i<M;++i)
        for (size_t j=0;j<N;++j)
            for (size_t k=0;k<K;++k)
                C(i,j) += A(i,k)*B(k,j);
    return C;
}

Matrix multiply_tile(const Matrix & A, const Matrix & B, size_t tsize)
{
    if (A.ncol() != B.nrow()) throw std::invalid_argument("dimension mismatch");
    if (tsize==0) return multiply_naive(A,B);
    const size_t M=A.nrow(), K=A.ncol(), N=B.ncol();
    Matrix C(M,N);
    for (size_t i0=0;i0<M;i0+=tsize) {
        size_t imax=std::min(i0+tsize,M);
        for (size_t j0=0;j0<N;j0+=tsize) {
            size_t jmax=std::min(j0+tsize,N);
            for (size_t k0=0;k0<K;k0+=tsize) {
                size_t kmax=std::min(k0+tsize,K);
                for (size_t i=i0;i<imax;++i)
                    for (size_t k=k0;k<kmax;++k) {
                        double a=A(i,k);
                        for (size_t j=j0;j<jmax;++j) C(i,j)+=a*B(k,j);
                    }
            }
        }
    }
    return C;
}

Matrix multiply_mkl(const Matrix & A, const Matrix & B)
{
    if (A.ncol() != B.nrow()) throw std::invalid_argument("dimension mismatch");
    Matrix C(A.nrow(),B.ncol());
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
        A.nrow(),B.ncol(),A.ncol(),
        1.0,A.data(),A.ncol(),B.data(),B.ncol(),
        0.0,C.data(),C.ncol());
    return C;
}