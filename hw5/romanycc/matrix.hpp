#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>

class Matrix {
public:
    Matrix(size_t nrow, size_t n_col) : rows_(nrow), cols_(n_col), data_(nrow * n_col, 0.0) {}

    double operator()(size_t r, size_t c) const {
        return data_[r * cols_ + c];
    }

    double& operator()(size_t r, size_t c) {
        return data_[r * cols_ + c];
    }

    inline size_t nrow() const { return rows_; }
    inline size_t ncol() const { return cols_; }

    bool operator==(const Matrix &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        for (size_t i = 0; i < data_.size(); ++i) {
            if (std::abs(data_[i] - other.data_[i]) > 1e-9) {
                return false;
            }
        }
        return true;
    }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

private:
    size_t rows_;
    size_t cols_;
    std::vector<double> data_;
};

Matrix multiply_naive(const Matrix &A, const Matrix &B);
Matrix multiply_tile(const Matrix &A, const Matrix &B, int T = 32);
Matrix multiply_mkl(const Matrix &A, const Matrix &B);
