#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"

namespace py = pybind11;

std::vector<int> infer_shape(py::handle obj)
{
    std::vector<int> shape;

    py::handle current = obj;

    while (py::isinstance<py::sequence>(current) &&
           !py::isinstance<py::str>(current))
    {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(current);

        shape.push_back(static_cast<int>(seq.size()));

        if (seq.size() == 0)
            break;

        current = seq[0];
    }

    return shape;
}

void validate_shape_recursive(
    py::handle obj,
    const std::vector<int> &shape,
    int dim)
{
    bool is_sequence =
        py::isinstance<py::sequence>(obj) &&
        !py::isinstance<py::str>(obj);

    if (dim == static_cast<int>(shape.size()))
    {
        if (is_sequence)
            throw std::runtime_error("Ragged tensor: inconsistent nested list shape");

        return;
    }

    if (!is_sequence)
        throw std::runtime_error("Ragged tensor: inconsistent nested list shape");

    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);

    if (static_cast<int>(seq.size()) != shape[dim])
        throw std::runtime_error("Ragged tensor: inconsistent nested list shape");

    for (py::handle item : seq)
    {
        validate_shape_recursive(item, shape, dim + 1);
    }
}

void flatten_recursive(py::handle obj, std::vector<float> &flat)
{
    if (py::isinstance<py::sequence>(obj) &&
        !py::isinstance<py::str>(obj))
    {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);

        for (py::handle item : seq)
        {
            flatten_recursive(item, flat);
        }
    }
    else
    {
        flat.push_back(py::cast<float>(obj));
    }
}

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

        .def(py::init([](py::object obj)
                      {
                          std::vector<int> shape = infer_shape(obj);

                          validate_shape_recursive(obj, shape, 0);

                          std::vector<float> flat;
                          flatten_recursive(obj, flat);

                          if (shape.empty())
                              shape = {static_cast<int>(flat.size())};

                          return Tensor(flat, shape); }))

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
        .def("flatten", &Tensor::flatten)
        .def("transpose", &Tensor::transpose)

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
