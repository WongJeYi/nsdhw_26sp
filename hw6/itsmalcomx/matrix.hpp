#pragma once
#include <vector>
#include <stdexcept>
#include <cstddef>

class Matrix
{
public:
    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol), m_buffer(nrow * ncol, 0.0)
    {}
    Matrix(Matrix const &)                 = default;
    Matrix & operator=(Matrix const &)     = default;
    Matrix(Matrix &&) noexcept             = default;
    Matrix & operator=(Matrix &&) noexcept = default;
    ~Matrix() = default;

    double   operator()(size_t r, size_t c) const { return m_buffer[r*m_ncol+c]; }
    double & operator()(size_t r, size_t c)       { return m_buffer[r*m_ncol+c]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    bool operator==(Matrix const & o) const
    { return m_nrow==o.m_nrow && m_ncol==o.m_ncol && m_buffer==o.m_buffer; }

    double *       data()       { return m_buffer.data(); }
    double const * data() const { return m_buffer.data(); }

private:
    size_t              m_nrow, m_ncol;
    std::vector<double> m_buffer;
};

Matrix multiply_naive(Matrix const & A, Matrix const & B);
Matrix multiply_tile (Matrix const & A, Matrix const & B, size_t tsize);
Matrix multiply_mkl  (Matrix const & A, Matrix const & B);