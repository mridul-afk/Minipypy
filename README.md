# MiniPyPy

> A PyTorch-inspired tensor framework built entirely from scratch using **C++**, **CUDA**, and **pybind11**.

**Current Version:** `v0.3.0`

MiniPyPy is an educational deep learning framework whose purpose is to understand how modern frameworks such as PyTorch work internally by implementing every major subsystem from first principles.

Rather than wrapping existing libraries, MiniPyPy focuses on building the core runtime—from tensor storage and CUDA kernels to memory management, automatic differentiation, and eventually neural network execution.

---

# Philosophy

Modern deep learning frameworks hide enormous engineering complexity behind a simple Python API.

MiniPyPy aims to expose that complexity by implementing every major component manually, allowing developers to understand:

- Tensor memory layout
- CUDA kernel execution
- GPU memory management
- Python ↔ C++ bindings
- Automatic differentiation
- Neural network execution

The long-term goal is for MiniPyPy to become the computational backend powering **TensorFold**, a tensor-decomposition-based deep learning framework.

---

# Features

## Tensor System

- Custom Tensor class
- N-dimensional tensor creation
- Shape metadata
- Stride computation
- GPU-first tensor storage
- CPU ↔ GPU memory transfers

## CUDA Backend

- CUDA-accelerated element-wise operations
- General 2D matrix multiplication
- Asynchronous CUDA kernel execution
- Custom CUDA memory pool
- Automatic GPU memory reuse

## Python Integration

- Python bindings using pybind11
- Native Python API

---

# Example

```python
import minipypy as mini

a = mini.Tensor(
    [1, 2,
     3, 4],
    [2, 2]
)

b = mini.Tensor(
    [5, 6,
     7, 8],
    [2, 2]
)

c = a + b

print(c.cpu())
```

---

# Current Capabilities

- ✅ N-dimensional tensor creation
- ✅ Shape & stride metadata
- ✅ CUDA element-wise operations
- ✅ General 2D matrix multiplication
- ✅ CUDA memory pooling
- ✅ Automatic GPU memory reuse
- ✅ Shape validation
- ✅ Shape preservation
- ✅ Move semantics
- ✅ Python bindings via pybind11

---

# Project Structure

```
MiniPyPy
│
├── src/
│   ├── bindings.cpp
│   ├── tensor.cpp
│   ├── tensor.cu
│   ├── memory_pool.cpp
│   ├── memory_pool.h
│   └── tensor.h
│
├── minipypy/
│
├── tests/
│
├── CMakeLists.txt
│
└── pyproject.toml
```

---

# Roadmap

## v0.1.0

- Tensor class
- CUDA backend
- Python bindings

---

## v0.2.0

- N-dimensional tensor support
- General 2D matrix multiplication
- Shape & stride metadata

---

## v0.3.0

### Performance & Memory Management

- CUDA memory pool
- GPU memory reuse
- Shape validation
- Shape preservation
- Asynchronous CUDA execution

---

## v0.4.0

### Tensor Operations

- Broadcasting
- Tensor reshaping
- Tensor transpose
- Tensor indexing
- Tensor slicing

---

## v0.5.0

### Automatic Differentiation

- Computational graph
- Backpropagation
- Gradient accumulation

---

## v0.6.0

### Neural Networks

- Linear layers
- Activation functions
- Loss functions
- Optimizers

---

## v1.0.0

### TensorFold Integration

- Tensor decomposition layers
- Low-rank tensor operations
- Model compression support
- TensorFold backend

---

# Building from Source

## Clone the repository

```bash
git clone https://github.com/mridul-afk/Minipypy.git
cd Minipypy
```

## Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -e .
```

## Build the CUDA backend

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release

cmake --install build --config Release
```

---

# Tech Stack

- C++17
- CUDA
- pybind11
- CMake
- Python

---

# Why MiniPyPy?

Deep learning frameworks are among the most sophisticated pieces of software engineering today.

MiniPyPy is an attempt to understand them by building every important subsystem from scratch rather than treating them as black boxes.

By the end of the project, MiniPyPy will include:

- Tensor runtime
- CUDA backend
- Memory allocator
- Automatic differentiation
- Neural network execution
- Tensor decomposition support

---

# Vision

MiniPyPy is **not** intended to replace PyTorch.

Instead, it serves as a learning-oriented framework for exploring how modern tensor libraries are engineered internally.

The ultimate objective is to develop a complete deep learning framework from first principles and use MiniPyPy as the computational backend for **TensorFold**, enabling efficient experimentation with tensor decomposition techniques, model compression, and high-performance deep learning systems.

---

## License

This project is licensed under the MIT License.
