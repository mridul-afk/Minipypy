# MiniPyPy

A minimal PyTorch-inspired tensor framework built from scratch using:

- C++
- CUDA
- PyBind11

## Current Features

- Custom Tensor class
- GPU memory management
- CUDA accelerated tensor operations
- Python bindings
- Tensor addition
- Tensor subtraction
- Tensor multiplication
- Tensor division
- CPU ↔ GPU transfer

## Example

```python
import minipypy as mini

a = mini.Tensor([1,2,3])
b = mini.Tensor([10,20,30])

c = a + b

print(c.cpu())
