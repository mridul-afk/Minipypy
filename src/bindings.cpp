#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <vector>

#include "tensor.h"

namespace py = pybind11;

std::string indent(int spaces)
{
    return std::string(spaces, ' ');
}

std::string format_tensor_recursive(
    const std::vector<float> &data,
    const std::vector<int> &shape,
    const std::vector<int> &stride,
    int dim,
    int offset,
    int indent_level)
{
    std::ostringstream oss;

    if (dim == static_cast<int>(shape.size()) - 1)
    {
        oss << "[";

        for (int i = 0; i < shape[dim]; i++)
        {
            oss << data[offset + i * stride[dim]];

            if (i != shape[dim] - 1)
                oss << ", ";
        }

        oss << "]";
        return oss.str();
    }

    oss << "[";

    for (int i = 0; i < shape[dim]; i++)
    {
        if (i != 0)
            oss << "\n"
                << indent(indent_level + 1);

        oss << format_tensor_recursive(
            data,
            shape,
            stride,
            dim + 1,
            offset + i * stride[dim],
            indent_level + 1);

        if (i != shape[dim] - 1)
            oss << ",";
    }

    oss << "]";

    return oss.str();
}

PYBIND11_MODULE(_C, m)
{
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<float> &>())
        .def(py::init<const std::vector<float> &, std::vector<int>>())

        .def("cpu", &Tensor::cpu)
        .def("shape", &Tensor::get_shape)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)

        .def("__add__", &Tensor::operator+)
        .def("__mul__", &Tensor::operator*)
        .def("__sub__", &Tensor::operator-)
        .def("__truediv__", &Tensor::operator/)

        .def("matmul", &Tensor::matmul)
        .def("__matmul__", &Tensor::matmul)
        .def("reshape", &Tensor::reshape)

        .def("__repr__", [](const Tensor &t)
             {
                 std::vector<float> data = t.cpu();
                 std::vector<int> shape = t.get_shape();
                 std::vector<int> stride = t.get_stride();

                 std::ostringstream oss;

                 oss << "Tensor(";

                 if (shape.empty())
                 {
                     oss << "[]";
                 }
                 else
                 {
                     oss << format_tensor_recursive(data, shape, stride, 0, 0, 0);
                 }

                 oss << ", shape=[";

                 for (size_t i = 0; i < shape.size(); i++)
                 {
                     oss << shape[i];

                     if (i != shape.size() - 1)
                         oss << ", ";
                 }

                 oss << "], device='cuda')";

                 return oss.str(); });
}
