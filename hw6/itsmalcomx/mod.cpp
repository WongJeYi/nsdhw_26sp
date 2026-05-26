#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_matrix, mod)
{
    py::class_<Matrix>(mod, "Matrix")
        .def(py::init<size_t, size_t>())
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("__eq__",
            [](Matrix const & s, Matrix const & o){ return s==o; })
        .def("__getitem__",
            [](Matrix const & s, std::tuple<size_t,size_t> idx)
            { return s(std::get<0>(idx),std::get<1>(idx)); })
        .def("__setitem__",
            [](Matrix & s, std::tuple<size_t,size_t> idx, double v)
            { s(std::get<0>(idx),std::get<1>(idx))=v; })
        .def_property("array",
            [](py::object self) -> py::array_t<double> {
                Matrix & m = self.cast<Matrix&>();
                return py::array_t<double>(
                    {(py::ssize_t)m.nrow(), (py::ssize_t)m.ncol()},
                    {(py::ssize_t)(m.ncol()*sizeof(double)), (py::ssize_t)sizeof(double)},
                    m.data(),
                    self   // keep Matrix alive as long as array lives
                );
            },
            nullptr)
    ;

    mod.def("multiply_naive", &multiply_naive);
    mod.def("multiply_tile",  &multiply_tile);
    mod.def("multiply_mkl",   &multiply_mkl);
}