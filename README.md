# MiniPyPy

MiniPyPy is a PyTorch-inspired tensor framework built entirely from scratch using **C++**, **CUDA**, and **pybind11**.

The goal of this project is to understand how modern deep learning frameworks such as PyTorch work internally by implementing every major component from first principles.

---

# Features

## Tensor System

* Custom Tensor class
* N-dimensional tensor creation
* Nested Python list support
* Shape & stride metadata
* GPU-first tensor storage
* CPU ↔ GPU memory transfers
* Shape validation
* Ragged tensor detection

## Tensor Operations

* Element-wise addition
* Element-wise subtraction
* Element-wise multiplication
* Element-wise division
* Broadcasting (NumPy/PyTorch style)
* 2D Matrix Multiplication
* Reshape
* Flatten
* 2D Transpose

## Reduction Operations

* Sum
* Mean
* Max

## CUDA Backend

* CUDA-accelerated tensor operations
* Asynchronous kernel execution
* Custom CUDA memory pool
* GPU memory reuse
* Broadcast-aware CUDA kernels

## Python Integration

* Python bindings using pybind11
* Natural nested-list tensor construction
* Readable N-dimensional tensor printing

---

# Example

```python
import minipypy as mini

a = mini.Tensor([
    [1, 2, 3],
    [4, 5, 6]
])

b = mini.Tensor([10, 20, 30])

print(a + b)

print(a.matmul(
    mini.Tensor([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
))

print(a.sum())
print(a.mean())
print(a.max())

print(a.reshape([3, 2]))
print(a.flatten())
print(a.transpose())
```

---

# Current Capabilities

* ✅ N-dimensional tensors
* ✅ Nested-list tensor construction
* ✅ Shape & stride management
* ✅ Broadcasting
* ✅ Element-wise tensor operations
* ✅ 2D matrix multiplication
* ✅ Reshape
* ✅ Flatten
* ✅ 2D transpose
* ✅ Tensor reductions
* ✅ CUDA memory pooling
* ✅ Automatic GPU memory reuse
* ✅ Shape validation
* ✅ Ragged tensor validation
* ✅ Move semantics
* ✅ Readable tensor representation

---

# Roadmap

## v0.1.0

* Tensor class
* CUDA backend
* Python bindings

## v0.2.0

* N-dimensional tensors
* General 2D matrix multiplication
* Shape & stride metadata

## v0.3.0

* CUDA memory pool
* Shape validation
* Shape preservation
* Asynchronous CUDA execution

## v0.4.0

* Broadcasting
* Reshape
* Flatten
* 2D Transpose
* Sum / Mean / Max reductions
* Nested-list tensor construction
* N-dimensional tensor printing
* Ragged tensor validation

## v0.5.0

* Automatic differentiation (Autograd)
* Computational graph
* Gradient propagation

## v0.6.0

* Neural network layers
* Activation functions
* Loss functions
* Optimizers

## v1.0.0

* TensorFold integration
* Tensor decomposition layers
* Model compression
* Training support

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

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

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

* C++17
* CUDA
* pybind11
* CMake
* Python

---

# Vision

MiniPyPy is an educational deep learning framework designed to explore how modern tensor libraries work internally.

Rather than wrapping existing frameworks, MiniPyPy implements tensor operations, CUDA kernels, memory management, broadcasting, reductions, and eventually automatic differentiation completely from scratch.

The long-term goal is to serve as the computational backend for **TensorFold**, a tensor-decomposition-based deep learning framework focused on efficient model compression and deployment.
