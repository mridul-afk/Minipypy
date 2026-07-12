# MiniPyPy

MiniPyPy is a minimal PyTorch-like deep learning framework built with a custom C++/CUDA tensor engine, Python bindings, reverse-mode autograd, and a growing high-level neural network API.

The goal of MiniPyPy is to build a small but understandable CUDA-backed deep learning framework from scratch, then extend it with TensorFold low-rank neural network layers.

## Current Status

MiniPyPy currently supports:

* CUDA-backed tensors
* Python bindings through pybind11
* Elementwise tensor operations
* Broadcasting
* Scalar tensor operations
* Reverse-mode autograd
* Matmul forward/backward
* ReLU forward/backward
* Basic neural network modules
* Sequential models
* SGD optimizer
* MSELoss
* HingeLoss

Latest milestone:

```text
v0.8.2 — Scalar Autograd and HingeLoss
```

## Example

```python
import minipypy as mini

x = mini.Tensor([[1.0], [2.0], [3.0], [4.0]])
y = mini.Tensor([[2.0], [4.0], [6.0], [8.0]])

model = mini.nn.Sequential(
    mini.nn.Linear(1, 4),
    mini.nn.ReLU(),
    mini.nn.Linear(4, 1),
)

loss_fn = mini.nn.MSELoss()
optimizer = mini.optim.SGD(model, lr=0.01)

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", loss_fn(model(x), y))
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/YOUR_USERNAME/minipypy.git
cd minipypy
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install build dependencies:

```powershell
pip install -U pip
pip install -e .
```

Or use the project build script:

```powershell
.\build.ps1 -sync
```

## Requirements

MiniPyPy currently requires:

* Python 3.11
* CMake
* Ninja
* Visual Studio Build Tools on Windows
* NVIDIA CUDA Toolkit
* pybind11
* scikit-build-core

The current development setup uses CUDA 13.0.

## Tensor API

Create tensors:

```python
import minipypy as mini

x = mini.Tensor([1.0, 2.0, 3.0])
y = mini.Tensor([10.0, 20.0, 30.0])
```

Elementwise operations:

```python
z = x + y
z = x - y
z = x * y
z = x / y
```

Scalar operations:

```python
z = x + 2.0
z = 2.0 + x

z = x - 2.0
z = 2.0 - x

z = x * 0.5
z = 0.5 * x

z = x / 2.0
z = 8.0 / x
```

Move result to CPU:

```python
print(z.cpu())
```

## Autograd

MiniPyPy supports reverse-mode autograd.

```python
x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x * x
loss = y.sum()

loss.backward()

print(x.grad().cpu())
```

Expected output:

```text
[2.0, 4.0, 6.0]
```

Scalar operations also support autograd:

```python
x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x * 0.5
loss = y.sum()

loss.backward()

print(x.grad().cpu())
```

Expected output:

```text
[0.5, 0.5, 0.5]
```

## Matmul

```python
a = mini.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = mini.Tensor([[10.0], [20.0]], requires_grad=True)

c = a @ b
loss = c.sum()

loss.backward()

print(a.grad())
print(b.grad())
```

MiniPyPy supports 2D matmul and N-D broadcasted batched matmul.

## ReLU

ReLU is implemented as a CUDA-backed operation with autograd support.

```python
x = mini.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

y = x.relu()
loss = y.sum()

loss.backward()

print(y.cpu())
print(x.grad().cpu())
```

Expected output:

```text
[0.0, 0.0, 0.0, 1.0, 2.0]
[0.0, 0.0, 0.0, 1.0, 1.0]
```

At `x = 0`, MiniPyPy uses gradient `0`.

## Neural Network API

MiniPyPy includes a small `mini.nn` package.

Currently supported:

```text
mini.nn.Module
mini.nn.Linear
mini.nn.ReLU
mini.nn.Sequential
mini.nn.MSELoss
mini.nn.HingeLoss
```

Functional API:

```text
mini.nn.functional.mse_loss
mini.nn.functional.relu
mini.nn.functional.hinge_loss
```

Optimizer API:

```text
mini.optim.SGD
```

## Linear Layer

```python
layer = mini.nn.Linear(3, 2)

x = mini.Tensor([[1.0, 2.0, 3.0]])

out = layer(x)
```

Internally:

```text
out = x @ W + b
```

## Sequential

```python
model = mini.nn.Sequential(
    mini.nn.Linear(1, 4),
    mini.nn.ReLU(),
    mini.nn.Linear(4, 1),
)

out = model(x)
```

Sequential supports:

```python
model(x)
model.parameters()
model.zero_grad()
model.step(lr)
len(model)
model[index]
```

## MSELoss

```python
loss_fn = mini.nn.MSELoss()

pred = model(x)
loss = loss_fn(pred, y)
```

Functional version:

```python
loss = mini.nn.functional.mse_loss(pred, y)
```

## HingeLoss

HingeLoss is useful for binary classification-style objectives.

Targets should be `-1` or `+1`.

```python
pred = mini.Tensor([[2.0], [-1.0], [0.5]], requires_grad=True)
target = mini.Tensor([[1.0], [-1.0], [1.0]])

loss = mini.nn.functional.hinge_loss(pred, target)

loss.backward()

print(loss)
print(pred.grad())
```

Formula:

```text
loss = mean(relu(1 - target * pred))
```

Module version:

```python
loss_fn = mini.nn.HingeLoss()
loss = loss_fn(pred, target)
```

## Optimizer

MiniPyPy currently supports SGD:

```python
optimizer = mini.optim.SGD(model, lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Currently, `SGD` takes the model object directly because parameter updates replace tensors rather than mutating them in-place.

## Tests

Run the full test suite:

```powershell
python -m pytest tests/test_autograd.py tests/test_scalar_ops.py tests/test_training.py tests/test_nn.py tests/test_relu.py tests/test_sequential.py tests/test_optim.py tests/test_scalar_autograd.py tests/test_hinge_loss.py -v
```

Expected result for v0.8.2:

```text
51 passed
```

## Project Roadmap

Near-term roadmap:

```text
v0.8.4 — CrossEntropyLoss
v0.8.5 — sigmoid / exp / log / sqrt CUDA primitives
v0.8.6 — Adam optimizer
v0.9.0 — Train XOR / tiny MLP demo
v0.10.0 — TensorFoldLinear prototype
```

Long-term goals:

* More tensor ops
* Better memory management
* In-place optimizer updates
* `no_grad()` context manager
* More activation functions
* More loss functions
* Convolution layers
* TensorFold low-rank layers
* CUDA kernel optimization
* cuBLAS/cuDNN integration

## TensorFold Direction

The long-term research goal is to integrate TensorFold layers into MiniPyPy.

Instead of only using normal dense layers like:

```text
y = xW + b
```

MiniPyPy will eventually support low-rank dense replacements:

```python
model = mini.nn.Sequential(
    mini.nn.TensorFoldLinear(784, 256, rank=8),
    mini.nn.ReLU(),
    mini.nn.TensorFoldLinear(256, 10, rank=4),
)
```

The goal is to reduce parameter count and memory usage while keeping training and inference GPU-backed.

## Known Limitations

MiniPyPy is still experimental.

Current limitations:

* No Adam optimizer yet
* No sigmoid, tanh, exp, log, or sqrt yet
* No convolution layers yet
* No TensorFold layers yet
* No `no_grad()` context manager yet
* Parameter updates currently replace tensors instead of mutating them in-place
* API and internals may change frequently

## License

Add your license here.

## Status

MiniPyPy is under active development.

Current milestone:

```text
Tensor + Autograd + mini.nn + ReLU + Sequential + SGD + HingeLoss
```
