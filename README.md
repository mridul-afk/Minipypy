# MiniPyPy

MiniPyPy is a minimal PyTorch-like deep learning framework built from scratch using a custom C++/CUDA tensor engine, Python bindings through pybind11, reverse-mode autograd, and a growing high-level neural network API.

The goal of MiniPyPy is to build a small but understandable CUDA-backed deep learning framework from scratch, then extend it with TensorFold compressed neural-network layers.

TensorFold is a subsystem inside MiniPyPy. It is not a separate framework.

---

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
* CUDA-backed `sqrt()`
* Fused CrossEntropyLoss forward/backward
* Fused BCEWithLogitsLoss forward/backward
* Basic neural network modules
* Sequential models
* SGD optimizer
* Adam optimizer
* MSELoss
* HingeLoss
* CrossEntropyLoss
* BCEWithLogitsLoss
* MNIST linear training examples
* TensorFoldLinear low-rank layer prototype
* TensorFoldLinear single-layer MNIST benchmark
* Dense MLP vs TensorFold MLP benchmark

Latest milestone:

```text
v0.9.0 — TensorFoldLinear Low-Rank Prototype
```

Full test suite:

```text
85 passed
```

---

## What TensorFold Currently Implements

TensorFold currently implements a first low-rank `TensorFoldLinear` prototype.

A normal dense Linear layer stores one full weight matrix:

```text
W: [in_features, out_features]
```

TensorFoldLinear replaces that matrix with two trainable low-rank factors:

```text
U: [in_features, rank]
V: [rank, out_features]
```

Dense Linear computes:

```text
Y = XW + b
```

TensorFoldLinear computes:

```text
Y = (XU)V + b
```

The full dense matrix `W` is not stored during normal forward execution.

This first implementation is **low-rank matrix factorization**.

MiniPyPy does **not** yet implement full tensor decomposition methods such as:

```text
CP decomposition
Tucker decomposition
Tensor Train decomposition
TT-SVD
HOSVD
```

Those are planned for future TensorFold versions.

---

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

---

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

---

## Quick Example

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

---

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

---

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

---

## Neural Network API

MiniPyPy includes a small `mini.nn` package.

Currently supported:

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

Optimizer API:

```text
mini.optim.SGD
mini.optim.Adam
```

---

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

where:

```text
W: [in_features, out_features]
b: [1, out_features]
```

---

## TensorFoldLinear Layer

TensorFoldLinear is MiniPyPy's first TensorFold layer prototype.

A normal dense Linear layer stores one full weight matrix:

```text
W: [in_features, out_features]
```

TensorFoldLinear replaces that matrix with two trainable low-rank factors:

```text
U: [in_features, rank]
V: [rank, out_features]
```

Dense Linear computes:

```text
Y = XW + b
```

TensorFoldLinear computes:

```text
Y = (XU)V + b
```

The full dense matrix `W` is never stored during normal forward execution.

Example:

```python
import minipypy as mini

layer = mini.nn.TensorFoldLinear(784, 10, rank=4)

x = mini.Tensor([[0.0 for _ in range(784)]])

out = layer(x)

print(out.shape())
print(layer.parameter_count())
print(layer.dense_parameter_count())
print(layer.compression_ratio())
```

For `TensorFoldLinear(784, 10, rank=4)`:

```text
TensorFold params = 784 × 4 + 4 × 10 + 10 = 3,186
Dense params      = 784 × 10 + 10 = 7,850
Compression       ≈ 2.46x
```

TensorFoldLinear uses Xavier initialization by default:

```python
layer = mini.nn.TensorFoldLinear(784, 10, rank=4)
```

This is equivalent to:

```python
layer = mini.nn.TensorFoldLinear(784, 10, rank=4, init="xavier")
```

A simple fixed-scale initialization is also available:

```python
layer = mini.nn.TensorFoldLinear(784, 10, rank=4, init="simple")
```

---

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

---

## Loss Functions

MiniPyPy currently supports:

```text
MSELoss
HingeLoss
CrossEntropyLoss
BCEWithLogitsLoss
```

### MSELoss

```python
loss_fn = mini.nn.MSELoss()

pred = model(x)
loss = loss_fn(pred, y)
```

Formula:

```text
loss = mean((pred - target)^2)
```

### HingeLoss

HingeLoss is useful for binary classification-style objectives.

Targets should be `-1` or `+1`.

```python
pred = mini.Tensor([[2.0], [-1.0], [0.5]], requires_grad=True)
target = mini.Tensor([[1.0], [-1.0], [1.0]])

loss = mini.nn.HingeLoss()(pred, target)

loss.backward()

print(loss)
print(pred.grad())
```

Formula:

```text
loss = mean(relu(1 - target * pred))
```

### CrossEntropyLoss

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

loss = mini.nn.CrossEntropyLoss()(logits, target)

loss.backward()

print(loss.cpu())
print(logits.grad().cpu())
```

MiniPyPy implements CrossEntropyLoss as a fused stable CUDA operation using:

```text
loss_i = -logit_correct + max_logit + log(sum_j exp(logit_j - max_logit))
```

The backward formula is:

```text
grad_logits = (softmax(logits) - one_hot(target)) / batch_size
```

### BCEWithLogitsLoss

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

loss = mini.nn.BCEWithLogitsLoss()(logits, target)

loss.backward()

print(loss.cpu())
print(logits.grad().cpu())
```

MiniPyPy implements BCEWithLogitsLoss as a fused stable CUDA operation using:

```text
loss = max(x, 0) - x * y + log(1 + exp(-abs(x)))
```

The backward formula is:

```text
grad_logits = (sigmoid(logits) - target) / num_elements
```

---

## Optimizers

MiniPyPy currently supports SGD and Adam.

### SGD

```python
optimizer = mini.optim.SGD(model, lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Adam

```python
optimizer = mini.optim.Adam(model, lr=0.001)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Adam keeps first and second moment estimates for each parameter:

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

Then applies bias correction:

```text
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

and updates parameters using:

```text
param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

Currently, optimizers take the model object directly because parameter updates replace tensors rather than mutating them in-place.

---

## MNIST Linear Classifier Example

MiniPyPy can train a simple MNIST linear classifier using its own tensor, autograd, loss, and optimizer stack.

Example model:

```python
model = mini.nn.Linear(784, 10)
loss_fn = mini.nn.CrossEntropyLoss()
optimizer = mini.optim.SGD(model, lr=0.1)
```

Adam can also be used:

```python
optimizer = mini.optim.Adam(model, lr=0.001)
```

Training pipeline:

```text
MNIST image
→ flatten to [784]
→ mini.Tensor
→ Linear(784, 10)
→ CrossEntropyLoss
→ backward
→ optimizer step
→ improved accuracy
```

Verified SGD run on a small MNIST subset:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = SGD
lr          = 0.1

epoch 1 summary: train_loss=1.1566 train_acc=0.7231 test_loss=0.7769 test_acc=0.8203
epoch 2 summary: train_loss=0.6286 train_acc=0.8442 test_loss=0.5910 test_acc=0.8477
epoch 3 summary: train_loss=0.5267 train_acc=0.8638 test_loss=0.5265 test_acc=0.8672
```

Verified Adam run on the same setup:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = Adam
lr          = 0.001

epoch 1 summary: train_loss=1.6186 train_acc=0.6592 test_loss=1.2126 test_acc=0.7734
epoch 2 summary: train_loss=0.9367 train_acc=0.8179 test_loss=0.8461 test_acc=0.8164
epoch 3 summary: train_loss=0.7047 train_acc=0.8501 test_loss=0.6977 test_acc=0.8262
```

---

## TensorFoldLinear Single-Layer MNIST Benchmark

This benchmark compares a dense single-layer MNIST classifier against TensorFoldLinear classifiers with different ranks.

Dense baseline:

```python
model = mini.nn.Linear(784, 10)
```

TensorFold variants:

```python
model = mini.nn.TensorFoldLinear(784, 10, rank=r, init="xavier")
```

Benchmark setup:

```text
batch_size  = 32
epochs      = 3
train_limit = 2048
test_limit  = 512
optimizer   = SGD
lr          = 0.1
init        = Xavier
```

Single-layer dense baseline:

```text
Dense Linear(784, 10)
Params:      7,850
Compression: 1.00x
Test Acc:    ~86.72%
```

TensorFoldLinear rank benchmark:

| Model                    | Params | Compression | Test Accuracy |
| ------------------------ | -----: | ----------: | ------------: |
| TensorFoldLinear rank=2  |  1,598 |       4.91x |        60.35% |
| TensorFoldLinear rank=4  |  3,186 |       2.46x |        76.95% |
| TensorFoldLinear rank=8  |  6,362 |       1.23x |        86.33% |
| TensorFoldLinear rank=10 |  7,950 |       0.99x |        86.91% |

On this small benchmark, `TensorFoldLinear rank=8` nearly matched the dense single-layer classifier while using fewer parameters.

`rank=10` reached slightly higher accuracy, but it is not a compression win because it uses more parameters than the dense layer.

---

## Dense MLP vs TensorFold MLP Benchmark

This benchmark compares a dense two-layer MLP against several TensorFold MLP variants.

Dense MLP:

```python
model = mini.nn.Sequential(
    mini.nn.Linear(784, 128),
    mini.nn.ReLU(),
    mini.nn.Linear(128, 10),
)
```

TensorFold MLP variants replace one or both dense layers with `TensorFoldLinear`.

Benchmark setup:

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

| Model                      |  Params | Compression | Test Accuracy |
| -------------------------- | ------: | ----------: | ------------: |
| Dense MLP                  | 101,770 |       1.00x |        85.55% |
| TensorFold MLP r16/r8      |  15,834 |       6.43x |        83.40% |
| TensorFold MLP r32/r8      |  30,426 |       3.34x |        83.79% |
| TensorFold MLP r32/r10     |  30,702 |       3.31x |        86.13% |
| TensorFold first layer r32 |  30,602 |       3.33x |        84.96% |

On this small MNIST benchmark, `TensorFold MLP r32/r10` reached dense-level accuracy while using about `3.31x` fewer parameters.

The key observation is:

```text
Compressing every layer is not always optimal.
Compressing large dense layers gives the best parameter savings.
Small final classifier layers may sometimes be better left dense or given higher rank.
```

---

## TensorFold Interpretation

The current TensorFoldLinear result demonstrates that MiniPyPy can train compressed low-rank neural networks from scratch.

The current workflow is:

```text
random factor initialization
        |
        v
train U and V directly
        |
        v
factorized forward pass
        |
        v
MiniPyPy autograd
        |
        v
SGD or Adam optimizer
```

This is different from pretrained compression.

MiniPyPy does not yet take an already-trained dense PyTorch model and compress it through SVD or tensor decomposition. That is future work.

Current workflow:

```text
Train factorized model from scratch
```

Future workflow:

```text
Take pretrained dense model
        |
        v
decompose dense weights
        |
        v
replace layers with TensorFold layers
        |
        v
fine-tune or run compressed inference
```

---

## Tests

Run the full test suite:

```powershell
python -m pytest tests/test_autograd.py tests/test_scalar_ops.py tests/test_training.py tests/test_nn.py tests/test_relu.py tests/test_sequential.py tests/test_optim.py tests/test_scalar_autograd.py tests/test_hinge_loss.py tests/test_softmax.py tests/test_cross_entropy_loss.py tests/test_bce_with_logits_loss.py tests/test_sqrt.py tests/test_adam.py tests/test_tensorfold_linear.py -v
```

Expected result for the current development snapshot:

```text
85 passed
```

---

## Project Roadmap

Current milestone:

```text
v0.9.0 — TensorFoldLinear low-rank prototype
```

Completed in this milestone:

```text
TensorFoldLinear layer
Xavier initialization
Parameter-count reporting
Compression-ratio reporting
Single-layer TensorFold MNIST benchmark
Dense MLP vs TensorFold MLP benchmark
```

Near-term next milestone:

```text
v0.10.0 — Tensorized / Tensor Decomposition Layer Prototype
```

Planned next work:

```text
- Improve TensorFold benchmark reproducibility
- Add more TensorFoldLinear tests
- Add gradient checks for U, V, and bias
- Explore true tensorized layers
- Begin Tensor Train Linear design
```

Later milestones:

```text
v0.11.0 — Dense-to-TensorFold conversion
v0.12.0 — SVD-based pretrained compression experiments
Later   — CP, Tucker, Tensor Train, CUDA optimizations
```

Long-term goals:

* More tensor ops
* Better memory management
* In-place optimizer updates
* `no_grad()` context manager
* More activation functions
* More loss functions
* Sigmoid, tanh, exp, and log primitive APIs
* Convolution layers
* TensorFold low-rank and tensor-decomposition layers
* Dense-to-TensorFold conversion utilities
* SVD-based compression experiments
* CUDA kernel optimization
* Possible cuBLAS/cuDNN integration
* Broadcast kernel metadata optimization using CUDA constant memory

---

## Future TensorFold Direction

The long-term TensorFold research goal is to integrate compressed and factorized neural-network layers directly into MiniPyPy.

The current first step is low-rank matrix factorization:

```text
W ≈ U @ V
```

Future TensorFold versions may support higher-order tensor decomposition methods.

A dense matrix:

```text
W: [in_features, out_features]
```

may be tensorized into a higher-order representation:

```text
W_tensor: [i1, i2, ..., id, o1, o2, ..., od]
```

where:

```text
in_features  = i1 × i2 × ... × id
out_features = o1 × o2 × ... × od
```

Possible future decomposition methods include:

```text
CP decomposition
Tucker decomposition
Tensor Train decomposition
```

Tensor Train is likely the best next decomposition target for neural-network layer compression.

---

## Future Optimization Notes

One possible CUDA optimization is to use constant memory for small read-only broadcasting metadata.

Current broadcast kernels pass shape and stride metadata through normal GPU memory.

Potential improvement:

```text
Store these in CUDA constant memory:
- output shape
- input shapes
- input strides
- ndim
```

This may improve broadcast kernels because all threads repeatedly read the same metadata values.

Tensor data itself should remain in global memory.

This optimization should be benchmarked later and is not required before TensorFold.

---

## Known Limitations

MiniPyPy is still experimental.

Current limitations:

* No convolution layers yet
* No full tensor decomposition layers yet; TensorFoldLinear is currently low-rank matrix factorization only
* No `no_grad()` context manager yet
* No true integer tensor dtype yet
* No sigmoid, tanh, exp, or log primitive APIs yet
* CrossEntropyLoss currently supports only `[batch, classes]` logits
* Parameter updates currently replace tensors instead of mutating them in-place
* Optimizer state is currently managed at the Python layer
* Some internal cloning paths still use CPU roundtrips and can be optimized later
* CUDA error handling is not yet centralized
* Memory pool does not yet support block splitting or stream-aware reuse
* API and internals may change frequently

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Status

MiniPyPy is under active development.

Current milestone:

```text
Tensor + Autograd + mini.nn + ReLU + Softmax + Sequential + SGD + Adam + HingeLoss + CrossEntropyLoss + BCEWithLogitsLoss + TensorFoldLinear + MNIST training
```
