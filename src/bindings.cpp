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
        validate_shape_recursive(item, shape, dim + 1);
}

void flatten_recursive(py::handle obj, std::vector<float> &flat)
{
    if (py::isinstance<py::sequence>(obj) &&
        !py::isinstance<py::str>(obj))
    {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);

        for (py::handle item : seq)
            flatten_recursive(item, flat);
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
        .def(py::init([](py::object obj, bool requires_grad)
                      {
                          std::vector<int> shape = infer_shape(obj);

                          validate_shape_recursive(obj, shape, 0);

                          std::vector<float> flat;
                          flatten_recursive(obj, flat);

                          if (shape.empty())
                              shape = {static_cast<int>(flat.size())};

                          return Tensor(flat, shape, requires_grad); }),
             py::arg("data"),
             py::arg("requires_grad") = false)

        .def(py::init([](const std::vector<float> &data,
                         std::vector<int> shape,
                         bool requires_grad)
                      { return Tensor(data, shape, requires_grad); }),
             py::arg("data"),
             py::arg("shape"),
             py::arg("requires_grad") = false)

        .def("cpu", &Tensor::cpu)
        .def("shape", &Tensor::get_shape)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)

        .def("grad", &Tensor::grad)
        .def("zero_grad", &Tensor::zero_grad)
        .def("backward", &Tensor::backward)

        /*
          Tensor-Tensor operators.
          These call the C++ Tensor operators, which create autograd nodes.
        */
        .def("__add__", [](const Tensor &a, const Tensor &b)
             { return a + b; }, py::is_operator())

        .def("__sub__", [](const Tensor &a, const Tensor &b)
             { return a - b; }, py::is_operator())

        .def("__mul__", [](const Tensor &a, const Tensor &b)
             { return a * b; }, py::is_operator())

        .def("__truediv__", [](const Tensor &a, const Tensor &b)
             { return a / b; }, py::is_operator())

        /*
          Autograd-safe scalar operators.

          Important:
          Do NOT call add_scalar/mul_scalar/sub_scalar/div_scalar here.
          Those methods are forward-only utility kernels and do not attach grad_fn.

          These overloads call:
              Tensor::operator+(float)
              Tensor::operator-(float)
              Tensor::operator*(float)
              Tensor::operator/(float)

          In tensor.cpp, those convert the scalar into Tensor([scalar]) and reuse
          tensor-tensor ops, so scalar expressions participate in autograd.
        */
        .def("__add__", [](const Tensor &a, float scalar)
             { return a + scalar; }, py::is_operator())

        .def("__radd__", [](const Tensor &a, float scalar)
             { return scalar + a; }, py::is_operator())

        .def("__sub__", [](const Tensor &a, float scalar)
             { return a - scalar; }, py::is_operator())

        .def("__rsub__", [](const Tensor &a, float scalar)
             { return scalar - a; }, py::is_operator())

        .def("__mul__", [](const Tensor &a, float scalar)
             { return a * scalar; }, py::is_operator())

        .def("__rmul__", [](const Tensor &a, float scalar)
             { return scalar * a; }, py::is_operator())

        .def("__truediv__", [](const Tensor &a, float scalar)
             { return a / scalar; }, py::is_operator())

        .def("__rtruediv__", [](const Tensor &a, float scalar)
             { return scalar / a; }, py::is_operator())

        .def("matmul", &Tensor::matmul)
        .def("__matmul__", &Tensor::matmul, py::is_operator())
        .def("reshape", &Tensor::reshape)
        .def("flatten", &Tensor::flatten)
        .def("transpose", &Tensor::transpose)
        .def("transpose_last_two_dims", &Tensor::transpose_last_two_dims)
        .def("sum", &Tensor::sum)
        .def("mean", &Tensor::mean)
        .def("max", &Tensor::max)
        .def("sum_to_shape", &Tensor::sum_to_shape)
        .def("clone", &Tensor::clone)

        /*
          Do not expose forward-only scalar helper methods such as:
              mul_scalar, add_scalar, sub_scalar, rsub_scalar, div_scalar, rdiv_scalar

          Users should use normal operators instead:
              x * 0.5
              2.0 - x
              8.0 / x

          Those operators are now autograd-safe.
        */
        .def("detach", &Tensor::detach)
        .def("requires_grad_", &Tensor::requires_grad_, py::arg("value") = true, py::return_value_policy::reference_internal)
        .def("relu", &Tensor::relu)

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
                     oss << format_tensor_recursive(
                         data,
                         shape,
                         stride,
                         0,
                         0,
                         0);
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
