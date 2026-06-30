# MiniPyPy

MiniPyPy is a PyTorch-inspired tensor framework built entirely from scratch using **C++**, **CUDA**, and **pybind11**.

The goal of this project is to understand how modern deep learning frameworks such as PyTorch work internally by implementing every major component from first principles.

---

## Features

### Tensor System

- Custom Tensor class
- N-dimensional tensor creation
- Shape & stride metadata
- GPU-first tensor storage
- CPU ↔ GPU memory transfers

### CUDA Backend

- CUDA-accelerated element-wise operations
- CUDA matrix multiplication (2D)
- Asynchronous kernel execution
- Custom CUDA memory pool
- GPU memory reuse

### Python Integration

- Python bindings using pybind11
- Native Python API

---

## Example

```python
import minipypy as mini

a = mini.Tensor(
    [1,2,3,4],
    [2,2]
)

b = mini.Tensor(
    [5,6,7,8],
    [2,2]
)

c = a + b

print(c.cpu())
```

---

## Current Capabilities

- ✅ N-dimensional tensor creation
- ✅ Shape & stride management
- ✅ Element-wise tensor operations
- ✅ 2D matrix multiplication
- ✅ CUDA memory pooling
- ✅ Automatic GPU memory reuse
- ✅ Shape validation
- ✅ Move semantics

---

## Roadmap

### v0.1.0

- Tensor class
- CUDA backend
- Python bindings

### v0.2.0

- N-dimensional tensor support
- General 2D matrix multiplication
- Shape & stride metadata

### v0.3.0

- CUDA memory pool
- Shape validation
- Shape preservation
- Asynchronous CUDA execution

### v0.4.0

- Broadcasting
- Tensor reshaping
- Tensor indexing
- Tensor transpose

### v0.5.0

- Automatic differentiation (Autograd)

### v0.6.0

- Neural network layers
- Optimizers
- Loss functions

### v1.0.0

- TensorFold integration
- Tensor decomposition layers
- Model compression support

---

## Building from Source

### Clone the repository

```bash
git clone https://github.com/mridul-afk/Minipypy.git
cd Minipypy
```

### Create a virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

### Install the package

```bash
pip install -e .
```

### Build the CUDA backend

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release
```

---

## Tech Stack

- C++17
- CUDA
- pybind11
- CMake
- Python

---

## Vision

MiniPyPy is not intended to replace PyTorch.

The objective is to build a deep learning framework from first principles, understand every layer of the stack—from CUDA kernels and tensor memory management to autograd and neural network execution—and ultimately serve as the backend for **TensorFold**, a tensor-decomposition-based deep learning framework.
