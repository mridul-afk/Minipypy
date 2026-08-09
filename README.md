# MiniPyPy

MiniPyPy is a minimal PyTorch-like deep learning framework built from scratch using a custom C++/CUDA tensor engine, Python bindings through pybind11, reverse-mode autograd, and a growing high-level neural-network API.

The goal is to build a small, understandable CUDA-backed deep-learning framework from scratch and then extend it with compressed and factorized neural-network layers through **TensorFold**.

**TensorFold is a subsystem inside MiniPyPy. It is not a separate framework.**

---

## Current Status

MiniPyPy currently supports:

* CUDA-backed tensors
* Python bindings through pybind11
* Elementwise tensor operations
* Broadcasting
* Scalar tensor operations with autograd
* Reverse-mode autograd
* 2D matrix multiplication
* N-D broadcasted batched matrix multiplication
* ReLU forward/backward
* N-D Softmax forward/backward
* CUDA-backed `sqrt()`
* Fused CrossEntropyLoss forward/backward
* Fused BCEWithLogitsLoss forward/backward
* Basic neural-network modules
* Sequential models
* SGD optimizer
* Adam optimizer
* MSELoss
* HingeLoss
* CrossEntropyLoss
* BCEWithLogitsLoss
* MNIST linear training examples
* TensorFoldLinear low-rank layer
* TensorFoldLinear single-layer MNIST benchmark
* Dense MLP vs TensorFold MLP benchmark
* Windows wheel builds
* Linux wheel builds
* Python 3.11–3.14 Windows wheels
* Python 3.12/3.14 Linux wheels

### Latest milestone

```text
v0.9.0 — TensorFoldLinear Low-Rank Prototype
```

### Test status

```text
92 passed
```

---

# Installation

MiniPyPy is currently distributed as platform-specific wheels through the project's CI builds.

Because MiniPyPy contains native C++/CUDA code, wheels are platform and Python-version specific.

## Windows

For a CPython 3.11 Windows environment:

```powershell
python -m venv venv
.\venv\Scripts\activate

pip install .\dist\minipypy-0.9.0-cp311-cp311-win_amd64.whl
```

Verify:

```powershell
python -c "import minipypy as mini; print(mini.__version__)"
```

Expected:

```text
0.9.0
```

The Windows wheel has been runtime-tested successfully with CUDA-backed tensors and autograd.

Example:

```powershell
python -c "import minipypy as mini; print(mini.Tensor([1.,2.,3.]))"
```

Expected:

```text
Tensor([1, 2, 3], shape=[3], device='cuda')
```

Autograd:

```powershell
python -c "import minipypy as mini; x=mini.Tensor([1.,2.,3.], requires_grad=True); y=(x*x).sum(); y.backward(); print(y); print(x.grad())"
```

Expected:

```text
Tensor([14], shape=[1], device='cuda')
Tensor([2, 4, 6], shape=[3], device='cuda')
```

---

## Linux / WSL

Linux wheels can be installed directly:

```bash
python -m venv minipypy-runtime-test
source minipypy-runtime-test/bin/activate

pip install /path/to/minipypy-0.9.0-cp312-cp312-linux_x86_64.whl
```

Verify:

```bash
python -c "import minipypy; print(minipypy.__file__); print(minipypy.__version__)"
```

Then:

```bash
python -c "import minipypy as mini; print(mini.Tensor([1.,2.,3.]))"
```

Expected:

```text
Tensor([1, 2, 3], shape=[3], device='cuda')
```

Autograd:

```bash
python -c "import minipypy as mini; x=mini.Tensor([1.,2.,3.], requires_grad=True); y=(x*x).sum(); y.backward(); print(y); print(x.grad())"
```

Expected:

```text
Tensor([14], shape=[1], device='cuda')
Tensor([2, 4, 6], shape=[3], device='cuda')
```

TensorFold:

```bash
python -c "import minipypy as mini; layer=mini.nn.TensorFoldLinear(4,3,rank=2); x=mini.Tensor([[1.,2.,3.,4.]]); print(layer(x)); print(layer.parameter_count())"
```

---

# Requirements

## Building from source

Building MiniPyPy from source currently requires:

* Python
* CMake
* Ninja
* C++ compiler
* NVIDIA CUDA Toolkit
* CUDA-compatible NVIDIA GPU/toolchain
* pybind11
* scikit-build-core

### Windows

The development configuration uses:

```text
Visual Studio
CMake
Ninja
NVIDIA CUDA Toolkit
```

The current local development environment uses CUDA 13.0.

### Linux

The CI build currently uses NVIDIA's CUDA development container:

```text
nvidia/cuda:13.3.1-devel-ubuntu24.04
```

---

# Runtime vs Build Requirements

MiniPyPy contains native CUDA code, so the requirements for **building** the framework and **running an already-built wheel** are different.

The CI wheels are compiled with CUDA support and contain the generated CUDA device code required by the MiniPyPy CUDA extension.

Therefore, the absence of `nvcc` on a runtime machine does not by itself mean that an already-built MiniPyPy wheel cannot run.

For example, the Windows runtime test successfully produced:

```text
Tensor([1, 2, 3], shape=[3], device='cuda')
```

and successfully executed CUDA-backed autograd even when:

```powershell
where.exe nvcc
```

did not find `nvcc`.

A compatible NVIDIA GPU and appropriate NVIDIA driver/runtime support are still required.

The exact runtime compatibility matrix will be expanded as more GPUs, drivers, CUDA versions, and operating systems are tested.

---

# Clone and Build From Source

Clone the repository:

```powershell
git clone https://github.com/mridul-afk/Minipypy.git
cd Minipypy
```

Create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install the project:

```powershell
pip install -U pip
pip install -e .
```

Or use the project build script:

```powershell
.\build.ps1 -sync
```

---

# Quick Example

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
optimizer = mini.optim.Adam(model, lr=0.05)

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Final loss:", loss_fn(model(x), y))
```

---

# Tensor API

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

Create N-D tensors:

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

Move results to CPU:

```python
print(z.cpu())
```

Inspect tensor metadata:

```python
print(x.shape())
print(x.ndim())
print(x.numel())
```

---

# Autograd

MiniPyPy supports reverse-mode autograd.

```python
x = mini.Tensor([1.0, 2.0, 3.0], requires_grad=True)

y = x * x
loss = y.sum()

loss.backward()

print(x.grad().cpu())
```

Expected:

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

Expected:

```text
[0.5, 0.5, 0.5]
```

Clear gradients:

```python
x.zero_grad()
```

Detach a tensor:

```python
y = x.detach()
```

Enable gradients:

```python
x = x.requires_grad_(True)
```

---

# Matrix Multiplication

```python
a = mini.Tensor(
    [[1.0, 2.0],
     [3.0, 4.0]],
    requires_grad=True
)

b = mini.Tensor(
    [[10.0],
     [20.0]],
    requires_grad=True
)

c = a @ b
loss = c.sum()

loss.backward()

print(a.grad())
print(b.grad())
```

MiniPyPy supports:

```text
2D matmul
N-D broadcasted batched matmul
```

---

# Neural Network API

MiniPyPy currently provides:

```text
mini.nn.Module
mini.nn.Linear
mini.nn.TensorFoldLinear
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

Optimizers:

```text
mini.optim.SGD
mini.optim.Adam
```

---

# Linear Layer

```python
layer = mini.nn.Linear(3, 2)

x = mini.Tensor([[1.0, 2.0, 3.0]])

out = layer(x)
```

Internally:

```text
out = x @ W + b
```

where:

```text
W: [in_features, out_features]
b: [1, out_features]
```

---

# TensorFoldLinear

TensorFold is currently implemented as a **low-rank matrix factorization system**.

A dense Linear layer stores:

```text
W: [in_features, out_features]
```

TensorFoldLinear instead stores two trainable factors:

```text
U: [in_features, rank]
V: [rank, out_features]
```

Dense computation:

```text
Y = XW + b
```

TensorFold computation:

```text
Y = (XU)V + b
```

The dense matrix `W` is not stored during normal TensorFoldLinear forward execution.

Conceptually:

```text
W ≈ U @ V
```

This reduces parameter count when the selected rank is sufficiently small.

---

## Example

```python
layer = mini.nn.TensorFoldLinear(
    784,
    10,
    rank=4
)

x = mini.Tensor([
    [0.0 for _ in range(784)]
])

out = layer(x)

print(out.shape())
print(layer.parameter_count())
print(layer.dense_parameter_count())
print(layer.compression_ratio())
```

For:

```text
TensorFoldLinear(784, 10, rank=4)
```

the parameter counts are:

```text
U = 784 × 4
V = 4 × 10
b = 10
```

Therefore:

```text
TensorFold parameters = 3,186
Dense parameters       = 7,850
```

giving approximately:

```text
2.46x parameter compression
```

---

## Initialization

TensorFoldLinear uses Xavier initialization by default:

```python
layer = mini.nn.TensorFoldLinear(
    784,
    10,
    rank=4
)
```

Equivalent to:

```python
layer = mini.nn.TensorFoldLinear(
    784,
    10,
    rank=4,
    init="xavier"
)
```

A simple fixed-scale initialization is also available:

```python
layer = mini.nn.TensorFoldLinear(
    784,
    10,
    rank=4,
    init="simple"
)
```

---

# Sequential

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

---

# Loss Functions

MiniPyPy currently supports:

```text
MSELoss
HingeLoss
CrossEntropyLoss
BCEWithLogitsLoss
```

## MSELoss

```python
loss_fn = mini.nn.MSELoss()

pred = model(x)
loss = loss_fn(pred, y)
```

Formula:

```text
loss = mean((pred - target)^2)
```

## HingeLoss

Targets should be:

```text
-1 or +1
```

```python
pred = mini.Tensor(
    [[2.0], [-1.0], [0.5]],
    requires_grad=True
)

target = mini.Tensor(
    [[1.0], [-1.0], [1.0]]
)

loss = mini.nn.HingeLoss()(pred, target)

loss.backward()

print(loss)
print(pred.grad())
```

Formula:

```text
loss = mean(relu(1 - target * pred))
```

## CrossEntropyLoss

Expected:

```text
logits: [batch, classes]
target: [batch] or [batch, 1]
```

Targets are class indices represented by float tensors.

```python
logits = mini.Tensor([
    [1.0, 2.0, 3.0],
    [3.0, 1.0, 2.0],
], requires_grad=True)

target = mini.Tensor([2.0, 0.0])

loss = mini.nn.CrossEntropyLoss()(logits, target)

loss.backward()

print(loss.cpu())
print(logits.grad().cpu())
```

MiniPyPy uses a fused stable CUDA implementation based on the log-sum-exp formulation.

Backward:

```text
grad_logits =
    (softmax(logits) - one_hot(target))
    / batch_size
```

## BCEWithLogitsLoss

Expected:

```text
logits.shape == target.shape
```

Targets must contain:

```text
0.0 or 1.0
```

Example:

```python
logits = mini.Tensor(
    [2.0, -1.0, 0.0],
    requires_grad=True
)

target = mini.Tensor(
    [1.0, 0.0, 1.0]
)

loss = mini.nn.BCEWithLogitsLoss()(logits, target)

loss.backward()

print(loss.cpu())
print(logits.grad().cpu())
```

Stable forward formulation:

```text
loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
```

Backward:

```text
grad_logits =
    (sigmoid(logits) - target)
    / num_elements
```

---

# Optimizers

MiniPyPy currently supports:

```text
SGD
Adam
```

## SGD

```python
optimizer = mini.optim.SGD(
    model,
    lr=0.01
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Adam

```python
optimizer = mini.optim.Adam(
    model,
    lr=0.001
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Adam uses first- and second-moment estimates:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t

v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
```

Bias correction:

```text
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

Parameter update:

```text
param =
    param -
    lr * m_hat / (sqrt(v_hat) + eps)
```

Currently, optimizer updates replace tensors rather than mutating them in-place.

---

# MNIST Linear Classifier

MiniPyPy can train a simple MNIST classifier using its own:

```text
Tensor
Autograd
Linear
CrossEntropyLoss
Optimizer
```

Example:

```python
model = mini.nn.Linear(784, 10)

loss_fn = mini.nn.CrossEntropyLoss()

optimizer = mini.optim.SGD(
    model,
    lr=0.1
)
```

Pipeline:

```text
MNIST image
     |
     v
flatten [784]
     |
     v
mini.Tensor
     |
     v
Linear(784, 10)
     |
     v
CrossEntropyLoss
     |
     v
backward()
     |
     v
optimizer.step()
```

Verified SGD run on a small MNIST subset:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = SGD
lr           = 0.1
```

Results:

```text
epoch 1:
train_loss=1.1566
train_acc=0.7231
test_loss=0.7769
test_acc=0.8203

epoch 2:
train_loss=0.6286
train_acc=0.8442
test_loss=0.5910
test_acc=0.8477

epoch 3:
train_loss=0.5267
train_acc=0.8638
test_loss=0.5265
test_acc=0.8672
```

Verified Adam run:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = Adam
lr          = 0.001
```

Results:

```text
epoch 1:
train_loss=1.6186
train_acc=0.6592
test_loss=1.2126
test_acc=0.7734

epoch 2:
train_loss=0.9367
train_acc=0.8179
test_loss=0.8461
test_acc=0.8164

epoch 3:
train_loss=0.7047
train_acc=0.8501
test_loss=0.6977
test_acc=0.8262
```

---

# TensorFold Benchmarks

## Single-Layer MNIST

Dense baseline:

```python
model = mini.nn.Linear(784, 10)
```

TensorFold:

```python
model = mini.nn.TensorFoldLinear(
    784,
    10,
    rank=r,
    init="xavier"
)
```

Benchmark configuration:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = SGD
lr          = 0.1
init        = Xavier
```

Results:

| Model              | Parameters | Compression | Test Accuracy |
| ------------------ | ---------: | ----------: | ------------: |
| Dense Linear       |      7,850 |       1.00x |       ~86.72% |
| TensorFold rank=2  |      1,598 |       4.91x |        60.35% |
| TensorFold rank=4  |      3,186 |       2.46x |        76.95% |
| TensorFold rank=8  |      6,362 |       1.23x |        86.33% |
| TensorFold rank=10 |      7,950 |       0.99x |        86.91% |

The important result is that:

```text
rank=8
```

nearly matched the dense classifier while using fewer parameters.

`rank=10` achieved slightly higher accuracy but is not a compression win because it contains more parameters than the dense model.

---

# Dense MLP vs TensorFold MLP

Dense model:

```python
model = mini.nn.Sequential(
    mini.nn.Linear(784, 128),
    mini.nn.ReLU(),
    mini.nn.Linear(128, 10),
)
```

TensorFold variants replace one or both dense layers.

Benchmark configuration:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = Adam
lr          = 0.001
init        = Xavier
```

Results:

| Model                      | Parameters | Compression | Test Accuracy |
| -------------------------- | ---------: | ----------: | ------------: |
| Dense MLP                  |    101,770 |       1.00x |        85.55% |
| TensorFold MLP r16/r8      |     15,834 |       6.43x |        83.40% |
| TensorFold MLP r32/r8      |     30,426 |       3.34x |        83.79% |
| TensorFold MLP r32/r10     |     30,702 |       3.31x |        86.13% |
| TensorFold first layer r32 |     30,602 |       3.33x |        84.96% |

On this small benchmark:

```text
TensorFold MLP r32/r10
```

achieved slightly higher test accuracy than the dense baseline while using approximately:

```text
3.31x fewer parameters
```

The main observation is:

```text
Compressing every layer is not always optimal.
```

Large dense layers provide more opportunities for parameter reduction, while small final classifier layers may be better kept dense or assigned a larger rank.

---

# TensorFold Interpretation

The current TensorFold implementation demonstrates that MiniPyPy can train compressed low-rank neural networks from scratch.

Current workflow:

```text
Random factor initialization
        |
        v
Train U and V directly
        |
        v
Factorized forward pass
        |
        v
MiniPyPy autograd
        |
        v
SGD / Adam
```

This is currently **training a factorized model from scratch**.

It is not yet pretrained-model compression.

The future workflow is:

```text
Pretrained dense model
        |
        v
Decompose dense weights
        |
        v
Replace dense layers
        |
        v
TensorFold layers
        |
        v
Fine-tune / compressed inference
```

---

# Current TensorFold Scope

TensorFold currently implements:

```text
Low-rank matrix factorization
```

Specifically:

```text
W ≈ U @ V
```

TensorFold does **not yet** implement:

```text
CP decomposition
Tucker decomposition
Tensor Train decomposition
TT-SVD
HOSVD
```

These are future research directions.

---

# Tests

Run the full test suite:

```powershell
python -m pytest -v
```

Current development snapshot:

```text
92 passed
```

The project also uses CI to build native wheels for Windows and Linux.

---

# CI / Wheel Builds

MiniPyPy currently has separate GitHub Actions workflows for:

```text
Windows wheels
Linux wheels
```

The Windows workflow builds multiple CPython versions:

```text
3.11
3.12
3.13
3.14
```

Linux builds currently cover:

```text
CPython 3.12
CPython 3.14
```

Each wheel is platform-specific.

Examples:

```text
minipypy-0.9.0-cp311-cp311-win_amd64.whl

minipypy-0.9.0-cp312-cp312-linux_x86_64.whl

minipypy-0.9.0-cp314-cp314-linux_x86_64.whl
```

The project is moving toward distributing prebuilt native wheels so users do not need to compile MiniPyPy themselves.

---

# Project Roadmap

## v0.9.0 — TensorFoldLinear Low-Rank Prototype

Completed:

```text
TensorFoldLinear layer
Xavier initialization
Parameter-count reporting
Compression-ratio reporting
Single-layer TensorFold MNIST benchmark
Dense MLP vs TensorFold MLP benchmark
Windows wheel CI
Linux wheel CI
Runtime wheel validation
```

## v0.10.0 — Tensorized / Tensor Decomposition Prototype

Planned:

```text
Improve TensorFold benchmark reproducibility
Add more TensorFoldLinear tests
Add gradient checks for U, V, and bias
Explore true tensorized layers
Begin Tensor Train Linear design
```

## v0.11.0

```text
Dense-to-TensorFold conversion
```

## v0.12.0

```text
SVD-based pretrained compression experiments
```

## Later

```text
CP decomposition
Tucker decomposition
Tensor Train
CUDA optimization
Compressed inference
```

---

# Long-Term Goals

MiniPyPy aims to grow into a compact CUDA-backed deep-learning framework with:

* More tensor operations
* Better memory management
* In-place optimizer updates
* `no_grad()` context manager
* More activation functions
* More loss functions
* Sigmoid
* Tanh
* Exp
* Log
* Convolution layers
* TensorFold low-rank layers
* TensorFold tensor-decomposition layers
* Dense-to-TensorFold conversion
* SVD-based pretrained-model compression
* CUDA kernel optimization
* Possible cuBLAS/cuDNN integration
* Better CUDA error handling
* Improved memory-pool management
* Compressed inference

---

# Future TensorFold Direction

The long-term TensorFold research goal is to integrate compressed and factorized neural-network layers directly into MiniPyPy.

The current starting point is:

```text
W ≈ U @ V
```

Future versions may tensorize a dense matrix:

```text
W: [in_features, out_features]
```

into a higher-order tensor:

```text
W_tensor:
[i1, i2, ..., id, o1, o2, ..., od]
```

where:

```text
in_features  = i1 × i2 × ... × id
out_features = o1 × o2 × ... × od
```

Potential decomposition methods:

```text
CP decomposition
Tucker decomposition
Tensor Train decomposition
```

Tensor Train is currently the most likely next major TensorFold decomposition target.

---

# Future CUDA Optimization

One possible CUDA optimization is moving small read-only broadcasting metadata into CUDA constant memory.

Potential metadata:

```text
output shape
input shapes
input strides
ndim
```

The motivation is that all threads repeatedly read the same metadata.

Tensor data itself should remain in global memory.

This optimization will be benchmarked independently and is not required for the current TensorFold milestone.

---

# Known Limitations

MiniPyPy is still experimental.

Current limitations include:

* No convolution layers yet
* TensorFold currently implements low-rank matrix factorization only
* No full CP/Tucker/Tensor Train implementation yet
* No `no_grad()` context manager
* No true integer tensor dtype yet
* No sigmoid, tanh, exp, or log primitive APIs yet
* CrossEntropyLoss currently supports `[batch, classes]` logits
* Parameter updates currently replace tensors instead of mutating them in-place
* Optimizer state is currently managed at the Python layer
* Some internal cloning paths still use CPU roundtrips
* CUDA error handling is not yet centralized
* Memory pool does not yet support block splitting or stream-aware reuse
* API and internals may change frequently
* Pretrained dense-model compression is not implemented yet
* CUDA/GPU/driver compatibility across all user environments has not yet been exhaustively tested

---

# Status

MiniPyPy is under active development.

Current milestone:

```text
v0.9.0 — TensorFoldLinear Low-Rank Prototype
```

Current stack:

```text
CUDA-backed Tensor
        +
C++ CUDA Engine
        +
pybind11
        +
Reverse-mode Autograd
        +
mini.nn
        +
SGD / Adam
        +
TensorFoldLinear
```

Current test status:

```text
92 passed
```

## The project is moving from a source-built experimental framework toward a **prebuilt-wheel experience**, with the long-term goal of making MiniPyPy usable by anyone with a compatible Python, NVIDIA GPU, and supported platform

# License

MiniPyPy is licensed under the MIT License.

See [`LICENSE`](LICENSE) for details.
