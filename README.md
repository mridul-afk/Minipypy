# MiniPyPy

MiniPyPy is a minimal PyTorch-like deep learning framework built with a custom C++/CUDA tensor engine, Python bindings, reverse-mode autograd, and a growing high-level neural network API.

The goal of MiniPyPy is to build a small but understandable CUDA-backed deep learning framework from scratch, then extend it with TensorFold low-rank neural network layers.

## Current Status

MiniPyPy currently supports:

* CUDA-backed tensors
* Python bindings through pybind11
* Elementwise tensor operations
* Broadcasting
* Scalar tensor operations with autograd
* Reverse-mode autograd
* 2D matmul forward/backward
* N-D broadcasted batched matmul forward/backward
* ReLU forward/backward
* N-D Softmax forward/backward
* Fused CrossEntropyLoss forward/backward
* Fused BCEWithLogitsLoss forward/backward
* Basic neural network modules
* Sequential models
* SGD optimizer
* MSELoss
* HingeLoss
* CrossEntropyLoss
* BCEWithLogitsLoss

Latest milestone:

```text
v0.8.5 — BCEWithLogitsLoss
```

Full test suite:

```text
75 passed
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
git clone https://github.com/mridul-afk/Minipypy.git
cd Minipypy
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install in editable mode:

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

Create tensors with gradients:

```python
x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)
```

Create N-D tensors from nested Python lists:

```python
x = mini.Tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])
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

Inspect tensor metadata:

```python
print(x.shape())
print(x.ndim())
print(x.numel())
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

Clear gradients:

```python
x.zero_grad()
```

Detach a tensor from the graph:

```python
y = x.detach()
```

Enable gradients on a tensor:

```python
x = x.requires_grad_(True)
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

Example:

```python
a = mini.Tensor([
    [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
    ]
])

b = mini.Tensor([
    [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    ]
])

out = a @ b
print(out.shape())
```

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

## Softmax

Softmax is implemented as a CUDA-backed N-D operation with autograd support.

```python
x = mini.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

y = x.softmax(dim=1)

print(y.cpu())
```

Functional API:

```python
y = mini.nn.functional.softmax(x, dim=1)
```

Module API:

```python
softmax = mini.nn.Softmax(dim=1)
y = softmax(x)
```

Softmax supports positive and negative dimensions:

```python
x.softmax(dim=0)
x.softmax(dim=1)
x.softmax(dim=-1)
```

## Neural Network API

MiniPyPy includes a small `mini.nn` package.

Currently supported:

```text
mini.nn.Module
mini.nn.Linear
mini.nn.ReLU
mini.nn.Softmax
mini.nn.Sequential
mini.nn.MSELoss
mini.nn.HingeLoss
mini.nn.CrossEntropyLoss
mini.nn.BCEWithLogitsLoss
```

Functional API:

```text
mini.nn.functional.mse_loss
mini.nn.functional.relu
mini.nn.functional.softmax
mini.nn.functional.hinge_loss
mini.nn.functional.cross_entropy
mini.nn.functional.binary_cross_entropy_with_logits
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

Formula:

```text
loss = mean((pred - target)^2)
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

## CrossEntropyLoss

CrossEntropyLoss is used for multi-class classification.

Expected shapes:

```text
logits: [batch, classes]
target: [batch] or [batch, 1]
```

Targets are class indices stored as float tensors:

```python
target = mini.Tensor([2.0, 0.0, 1.0])
```

Example:

```python
logits = mini.Tensor([
    [1.0, 2.0, 3.0],
    [3.0, 1.0, 2.0],
], requires_grad=True)

target = mini.Tensor([2.0, 0.0])

loss = mini.nn.functional.cross_entropy(logits, target)

loss.backward()

print(loss.cpu())
print(logits.grad().cpu())
```

Module version:

```python
loss_fn = mini.nn.CrossEntropyLoss()
loss = loss_fn(logits, target)
```

MiniPyPy implements CrossEntropyLoss as a fused stable CUDA operation using:

```text
loss_i = -logit_correct + max_logit + log(sum_j exp(logit_j - max_logit))
```

The backward formula is:

```text
grad_logits = (softmax(logits) - one_hot(target)) / batch_size
```

## BCEWithLogitsLoss

BCEWithLogitsLoss is used for binary classification and multi-label classification.

Expected shape rule:

```text
logits.shape == target.shape
```

Targets must contain:

```text
0.0 or 1.0
```

Example:

```python
logits = mini.Tensor([2.0, -1.0, 0.0], requires_grad=True)
target = mini.Tensor([1.0, 0.0, 1.0])

loss = mini.nn.functional.binary_cross_entropy_with_logits(logits, target)

loss.backward()

print(loss.cpu())
print(logits.grad().cpu())
```

Module version:

```python
loss_fn = mini.nn.BCEWithLogitsLoss()
loss = loss_fn(logits, target)
```

MiniPyPy implements BCEWithLogitsLoss as a fused stable CUDA operation using:

```text
loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
```

The backward formula is:

```text
grad_logits = (sigmoid(logits) - target) / num_elements
```

Multi-label example:

```python
logits = mini.Tensor([
    [2.0, -1.0, 0.0],
    [-2.0, 3.0, 1.0],
], requires_grad=True)

target = mini.Tensor([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
])

loss = mini.nn.BCEWithLogitsLoss()(logits, target)
loss.backward()
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
python -m pytest tests/test_autograd.py tests/test_scalar_ops.py tests/test_training.py tests/test_nn.py tests/test_relu.py tests/test_sequential.py tests/test_optim.py tests/test_scalar_autograd.py tests/test_hinge_loss.py tests/test_softmax.py tests/test_cross_entropy_loss.py tests/test_bce_with_logits_loss.py -v
```

Expected result for v0.8.5:

```text
75 passed
```

## Project Roadmap

Near-term roadmap:

```text
v0.9.0 — MNIST Linear Classifier
v0.9.1 — Accuracy / argmax helper utilities
v0.9.2 — Better training examples and benchmarks
v0.10.0 — TensorFoldLinear prototype
```

Long-term goals:

* More tensor ops
* Better memory management
* In-place optimizer updates
* `no_grad()` context manager
* More activation functions
* More loss functions
* Adam optimizer
* Sigmoid, tanh, exp, log, and sqrt primitives
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
* No sigmoid, tanh, exp, log, or sqrt primitive APIs yet
* No convolution layers yet
* No TensorFold layers yet
* No `no_grad()` context manager yet
* No true integer tensor dtype yet
* CrossEntropyLoss currently supports only `[batch, classes]` logits
* Parameter updates currently replace tensors instead of mutating them in-place
* Some internal cloning paths still use CPU roundtrips and can be optimized later
* API and internals may change frequently

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Status

MiniPyPy is under active development.

Current milestone:

```text
Tensor + Autograd + mini.nn + ReLU + Softmax + Sequential + SGD + HingeLoss + CrossEntropyLoss + BCEWithLogitsLoss
```
