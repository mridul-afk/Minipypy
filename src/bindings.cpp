#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m)
{
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<float> &>())

        .def("cpu", &Tensor::cpu)

        .def("__add__", &Tensor::operator+)
        .def("__mul__", &Tensor::operator*)
        .def("__sub__", &Tensor::operator-)
        .def("__truediv__", &Tensor::operator/)
        .def("__repr__", [](const Tensor &t)
             {
    std::vector<float> data = t.cpu();

    std::string out = "Tensor([";
    for (size_t i = 0; i < data.size(); i++) {
        out += std::to_string(data[i]);
        if (i != data.size() - 1) out += ", ";
    }
    out += "], device='cuda')";
    return out; });
}
