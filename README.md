# MiniPyPy

MiniPyPy is a PyTorch-inspired tensor framework built from scratch using C++, CUDA, and pybind11.

The goal of this project is to understand and implement the core components of modern deep learning frameworks from first principles.

## Current Features

- Custom Tensor class
- CUDA-accelerated tensor operations
- Python bindings using pybind11
- CPU ↔ GPU memory transfers
- Element-wise tensor arithmetic

## Example

```python
import minipypy as mini

a = mini.Tensor([1, 2, 3])
b = mini.Tensor([10, 20, 30])

c = a + b

print(c.cpu())
```

## Roadmap

### v0.1

- Tensor class
- CUDA backend
- Python bindings

### v0.2

- Matrix multiplication
- Broadcasting

### v0.3

- Autograd engine

### v0.4

- Neural network layers

### v0.5

- TensorFold compression engine

## Installation

MiniPyPy is currently in early development.

### Install from source

```bash
git clone https://github.com/mridul-afk/Minipypy.git
cd MiniPyPy
pip install .
```
