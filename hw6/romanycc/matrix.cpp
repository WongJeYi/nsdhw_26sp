#include "matrix.hpp"
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <vector>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <mkl.h>
#endif

using namespace std;

Matrix multiply_naive(const Matrix &A, const Matrix &B) {
  if (A.ncol() != B.nrow()) {
    throw invalid_argument("Incompatible matrix dimensions for multiplication");
  }
  Matrix C(A.nrow(), B.ncol());
  for (size_t i = 0; i < A.nrow(); ++i) {
    for (size_t j = 0; j < B.ncol(); ++j) {
      double sum = 0;
      for (size_t k = 0; k < A.ncol(); ++k) {
        sum += A(i, k) * B(k, j);
      }
      C(i, j) = sum;
    }
  }
  return C;
}

Matrix multiply_tile(const Matrix &A, const Matrix &B, int T) {
  if (A.ncol() != B.nrow()) {
    throw invalid_argument("Incompatible matrix dimensions for multiplication");
  }
  Matrix C(A.nrow(), B.ncol());
  for (size_t i = 0; i < A.nrow(); i += T) {
    for (size_t j = 0; j < B.ncol(); j += T) {
      for (size_t k = 0; k < A.ncol(); k += T) {
        for (size_t ii = i; ii < min(i + (size_t)T, A.nrow()); ++ii) {
          for (size_t jj = j; jj < min(j + (size_t)T, B.ncol()); ++jj) {
            double sum = C(ii, jj);
            for (size_t kk = k; kk < min(k + (size_t)T, A.ncol()); ++kk) {
              sum += A(ii, kk) * B(kk, jj);
            }
            C(ii, jj) = sum;
          }
        }
      }
    }
  }
  return C;
}

Matrix multiply_mkl(const Matrix &A, const Matrix &B) {
  if (A.ncol() != B.nrow()) {
    throw invalid_argument("Incompatible matrix dimensions for multiplication");
  }

  Matrix C(A.nrow(), B.ncol());
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.nrow(), B.ncol(),
              A.ncol(), 1.0, A.data(), A.ncol(), B.data(), B.ncol(), 0.0,
              C.data(), C.ncol());

  return C;
}