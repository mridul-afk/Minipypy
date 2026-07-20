# MiniPyPy Architecture

MiniPyPy is a minimal PyTorch-inspired deep-learning framework built from scratch using C++, CUDA, Python, and pybind11.

The project is designed to make the internals of a deep-learning framework understandable while still providing GPU-backed tensor operations, automatic differentiation, neural-network modules, optimizers, and model-training examples.

TensorFold is a subsystem of MiniPyPy that will add compressed and factorized neural-network layers.

---

## 1. Design Goals

MiniPyPy has five primary goals:

1. Build a usable tensor engine using custom C++ and CUDA code.
2. Keep tensor data resident on the GPU during normal computation.
3. Provide a Python API similar to a small subset of PyTorch.
4. Implement reverse-mode automatic differentiation from scratch.
5. Extend the framework with TensorFold compressed neural-network layers.

MiniPyPy is primarily an educational and experimental framework. It is not intended to compete with the performance, hardware coverage, or API completeness of PyTorch or TensorFlow.

---

## 2. High-Level Architecture

MiniPyPy is divided into four major layers:

```text
User Python Code
       |
       v
MiniPyPy Python API
       |
       v
pybind11 C++ Bindings
       |
       v
C++ Tensor and Autograd Engine
       |
       v
Custom CUDA Kernels
       |
       v
NVIDIA GPU
```

The main components are:

```text
minipypy/
├── nn/                 Python neural-network API
├── optim/              Python optimizers
└── _C                  Compiled C++/CUDA extension

src/
├── tensor.h            Tensor class declaration
├── tensor.cpp          Tensor operations and graph construction
├── tensor.cu           CUDA forward kernels
├── autograd.h          Autograd node definitions
├── autograd.cpp        Backward graph traversal
├── autograd.cu         CUDA backward kernels
├── memory_pool.h       CUDA memory-pool interface
├── memory_pool.cpp     CUDA memory-pool implementation
└── bindings.cpp        Python bindings through pybind11
```

---

## 3. Python API Layer

The Python API provides the interface used by applications and training scripts.

Example:

```python
import minipypy as mini

x = mini.Tensor([[1.0, 2.0]], requires_grad=True)

model = mini.nn.Sequential(
    mini.nn.Linear(2, 4),
    mini.nn.ReLU(),
    mini.nn.Linear(4, 1),
)

prediction = model(x)
loss = prediction.mean()
loss.backward()
```

The Python layer currently contains:

```text
mini.Tensor

mini.nn.Module
mini.nn.Linear
mini.nn.ReLU
mini.nn.Softmax
mini.nn.Sequential

mini.nn.MSELoss
mini.nn.HingeLoss
mini.nn.CrossEntropyLoss
mini.nn.BCEWithLogitsLoss

mini.optim.SGD
mini.optim.Adam
```

Most tensor computation is delegated to the compiled `_C` extension.

The Python layer is mainly responsible for:

* Constructing neural-network modules
* Storing model parameters
* Organizing layers
* Providing loss-function wrappers
* Managing optimizer state
* Exposing a convenient user-facing API

---

## 4. Python-to-C++ Bindings

The file `src/bindings.cpp` uses pybind11 to expose the C++ `Tensor` class to Python.

It connects Python expressions such as:

```python
c = a + b
c = a @ b
c = a.relu()
c = a.softmax(dim=-1)
c.backward()
```

to C++ methods such as:

```cpp
Tensor::operator+()
Tensor::matmul()
Tensor::relu()
Tensor::softmax()
Tensor::backward()
```

The bindings also perform Python-specific work, including:

* Inferring tensor shape from nested Python lists
* Validating rectangular nested-list input
* Flattening nested input into contiguous storage
* Formatting tensors for `__repr__`
* Binding Python operators to C++ operators

Example flow:

```text
Python:
    z = x + y

pybind11:
    Tensor::__add__

C++:
    Tensor::operator+(const Tensor&)

CUDA:
    launch_add(...)
```

---

## 5. Tensor Representation

The central C++ class is `Tensor`.

Its important fields are:

```cpp
float* d_data;
int size;

std::vector<int> shape;
std::vector<int> stride;

bool is_cuda;
bool requires_grad;

Tensor* grad_tensor;
std::shared_ptr<AutogradNode> grad_fn;
```

### `d_data`

`d_data` points to the tensor's values stored in CUDA device memory.

MiniPyPy currently uses GPU-resident float tensors.

### `size`

`size` stores the total number of elements:

```text
size = product(shape)
```

For a tensor with shape:

```text
[2, 3, 4]
```

the size is:

```text
2 × 3 × 4 = 24
```

### `shape`

`shape` describes the dimensions of the tensor.

Examples:

```text
[4]       → vector
[2, 3]    → matrix
[2, 3, 4] → rank-3 tensor
```

### `stride`

MiniPyPy currently uses contiguous row-major storage.

For shape:

```text
[2, 3, 4]
```

the stride is:

```text
[12, 4, 1]
```

The flat offset of coordinate `[i, j, k]` is:

```text
offset = i × 12 + j × 4 + k × 1
```

### `requires_grad`

This flag determines whether gradients should be tracked for the tensor.

### `grad_tensor`

This stores the gradient accumulated during the backward pass.

### `grad_fn`

This points to the autograd node that created the tensor.

A leaf parameter normally has:

```text
requires_grad = true
grad_fn = null
```

An intermediate tensor normally has:

```text
grad_fn != null
```

---

## 6. GPU Memory Management

Tensor memory is allocated through a custom CUDA memory pool.

The memory pool is implemented in:

```text
src/memory_pool.h
src/memory_pool.cpp
```

Without a memory pool, tensor creation and destruction would repeatedly call:

```cpp
cudaMalloc(...)
cudaFree(...)
```

These operations can add significant overhead.

MiniPyPy instead reuses previously allocated memory blocks.

Conceptually:

```text
Tensor A requests 4096 bytes
        |
        v
Memory pool has no free block
        |
        v
cudaMalloc creates block
        |
        v
Tensor A uses block
        |
        v
Tensor A is destroyed
        |
        v
Block returns to memory pool
        |
        v
Tensor B requests 4096 bytes
        |
        v
Tensor B reuses the same block
```

The current pool groups free blocks by allocation size:

```cpp
std::unordered_map<size_t, std::vector<void*>> free_blocks;
```

When a tensor is destroyed, its memory is returned to the pool rather than immediately released to CUDA.

The pool releases cached blocks when the process exits.

---

## 7. Tensor Ownership and Move Semantics

A tensor directly owns GPU memory through `d_data`.

Copying a tensor object without carefully duplicating its device memory could cause:

* Double deallocation
* Multiple tensors owning the same pointer
* Corrupted gradients
* Invalid device-memory access

For this reason, normal copying is disabled:

```cpp
Tensor(const Tensor&) = delete;
Tensor& operator=(const Tensor&) = delete;
```

Move construction and move assignment are supported:

```cpp
Tensor(Tensor&& other) noexcept;
Tensor& operator=(Tensor&& other) noexcept;
```

Move semantics transfer ownership of the GPU pointer from one tensor object to another.

After the move:

```text
new tensor owns d_data
old tensor d_data becomes null
```

This is useful when tensor operations return temporary tensors.

---

## 8. CUDA Forward Operations

Forward CUDA kernels are primarily implemented in:

```text
src/tensor.cu
```

Current operations include:

* Elementwise addition
* Elementwise subtraction
* Elementwise multiplication
* Elementwise division
* Broadcasting
* Scalar arithmetic
* Matrix multiplication
* Batched matrix multiplication
* Broadcasted batched matrix multiplication
* Transpose
* Reduction sum
* Reduction mean
* ReLU
* Softmax
* Cross-entropy loss
* BCE-with-logits loss
* Square root

A typical elementwise kernel uses one CUDA thread per output element:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;

if (idx < size)
    output[idx] = a[idx] + b[idx];
```

---

## 9. Broadcasting

MiniPyPy supports NumPy-style broadcasting for compatible tensor shapes.

Two dimensions are compatible when:

```text
dimension_a == dimension_b
```

or one of them is:

```text
1
```

Example:

```text
[4, 3] + [1, 3] → [4, 3]
```

Broadcast shape computation is performed in C++.

The CUDA kernels map each output coordinate back to the correct offset in each input tensor.

For a broadcast dimension, the input coordinate is forced to zero:

```text
input dimension = 1
output coordinate = any value
input coordinate = 0
```

The backward pass uses `sum_to_shape()` to reduce the broadcasted gradient back to the original input shape.

Example:

```text
forward:
    [4, 3] + [1, 3] → [4, 3]

backward for second input:
    gradient [4, 3]
        |
        v
    sum over broadcast dimension
        |
        v
    gradient [1, 3]
```

---

## 10. Matrix Multiplication

MiniPyPy supports:

* 2D matrix multiplication
* Batched matrix multiplication
* Broadcasted batched matrix multiplication

For matrices:

```text
A shape = [M, K]
B shape = [K, N]
C shape = [M, N]
```

The mathematical operation is:

```text
C[i, j] = Σ A[i, k]B[k, j]
```

The CUDA implementation uses tiled matrix multiplication with shared memory.

Conceptually:

```text
Global-memory tile of A
        |
        v
Shared-memory tile of A

Global-memory tile of B
        |
        v
Shared-memory tile of B

Threads multiply shared tiles
        |
        v
Partial sums accumulate
        |
        v
Output matrix C
```

For N-dimensional inputs, the last two dimensions are treated as matrices and the preceding dimensions are treated as batch dimensions.

Example:

```text
A: [2, 4, 3, 5]
B: [1, 4, 5, 6]

Output:
[2, 4, 3, 6]
```

The batch dimensions are broadcast before performing each matrix multiplication.

---

## 11. Autograd Graph

MiniPyPy implements reverse-mode automatic differentiation.

Every differentiable operation creates an `AutogradNode`.

The node stores:

```cpp
OpType op;
std::vector<Tensor*> parents;
std::vector<std::shared_ptr<Tensor>> saved_tensors;
int dim;
```

### `op`

Identifies the operation that created the tensor.

Examples:

```text
ADD
SUB
MUL
DIV
MATMUL
SUM
MEAN
RELU
SOFTMAX
CROSS_ENTROPY
BCE_WITH_LOGITS
SQRT
```

### `parents`

Stores the tensors that were inputs to the operation.

### `saved_tensors`

Keeps temporary tensors alive when backward needs their values.

This is necessary for expressions such as:

```python
loss = (x * x).sum()
```

The intermediate result `x * x` may be a temporary Python object, but the backward pass still requires its graph and values.

### `dim`

Stores extra operation metadata.

For example, softmax backward needs to know the dimension over which softmax was applied.

---

## 12. Backward Pass

Calling:

```python
loss.backward()
```

invokes:

```cpp
Tensor::backward()
```

The backward process has four main stages.

### Stage 1: Build the topological order

MiniPyPy recursively traverses the graph from the output tensor to its parents.

```text
loss
 |
 v
operation
 |
 v
parents
 |
 v
leaf tensors
```

Each tensor is visited once.

### Stage 2: Initialize gradients

Gradient buffers are created and initialized to zero.

The output gradient is initialized to one:

```text
d(loss) / d(loss) = 1
```

### Stage 3: Reverse the topological order

Forward graph order:

```text
x → operation 1 → operation 2 → loss
```

Backward order:

```text
loss → operation 2 → operation 1 → x
```

### Stage 4: Execute backward rules

Each operation applies its derivative rule.

Examples:

```text
z = a + b

dz/da = 1
dz/db = 1
```

```text
z = a × b

dz/da = b
dz/db = a
```

```text
C = A @ B

dL/dA = dL/dC @ Bᵀ
dL/dB = Aᵀ @ dL/dC
```

Backward CUDA kernels are implemented in:

```text
src/autograd.cu
```

Graph traversal and operation dispatch are implemented in:

```text
src/autograd.cpp
```

---

## 13. Gradient Accumulation

A tensor may contribute to the output through multiple graph paths.

Example:

```python
y = x * x + x
```

The total derivative is:

```text
dy/dx = 2x + 1
```

Gradients must therefore accumulate instead of overwrite one another.

MiniPyPy backward kernels use addition or `atomicAdd` where necessary:

```cpp
grad_in[idx] += contribution;
```

This allows multiple backward paths to contribute to the same gradient buffer.

---

## 14. Neural-Network Modules

The Python module system is implemented in:

```text
minipypy/nn/modules.py
```

Every layer inherits from `Module`.

The base interface provides:

```python
forward()
parameters()
named_parameters()
zero_grad()
step()
```

### Linear Layer

A linear layer computes:

```text
output = input @ W + b
```

where:

```text
W shape = [in_features, out_features]
b shape = [1, out_features]
```

Bias addition uses broadcasting.

### Sequential

`Sequential` applies layers in order:

```python
model = mini.nn.Sequential(
    mini.nn.Linear(784, 128),
    mini.nn.ReLU(),
    mini.nn.Linear(128, 10),
)
```

Execution flow:

```text
input
  |
  v
Linear
  |
  v
ReLU
  |
  v
Linear
  |
  v
output
```

---

## 15. Loss Functions

MiniPyPy currently provides:

### Mean Squared Error

```text
MSE = mean((prediction - target)²)
```

### Hinge Loss

```text
HingeLoss = mean(max(0, 1 - target × prediction))
```

### Cross-Entropy Loss

For class-index targets:

```text
loss_i =
-logit[target]
+ max(logits)
+ log(Σ exp(logit - max(logits)))
```

This fused formulation is numerically stable.

Its gradient is:

```text
grad_logits =
(softmax(logits) - one_hot(target)) / batch_size
```

### BCE With Logits

The stable forward formula is:

```text
max(x, 0) - x × y + log(1 + exp(-|x|))
```

The gradient is:

```text
(sigmoid(x) - y) / number_of_elements
```

---

## 16. Optimizers

MiniPyPy currently supports SGD and Adam.

### SGD

```text
parameter = parameter - learning_rate × gradient
```

### Adam

Adam tracks first and second moments:

```text
m_t = β₁m_{t-1} + (1 - β₁)g_t
v_t = β₂v_{t-1} + (1 - β₂)g_t²
```

Bias correction:

```text
m_hat = m_t / (1 - β₁ᵗ)
v_hat = v_t / (1 - β₂ᵗ)
```

Parameter update:

```text
parameter =
parameter - learning_rate × m_hat / (sqrt(v_hat) + epsilon)
```

Optimizer state is currently managed in Python.

Parameters are updated by creating replacement tensors rather than modifying tensor storage in place.

---

## 17. End-to-End Training Flow

A MiniPyPy training iteration follows this path:

```text
Python training script
        |
        v
Model forward()
        |
        v
Linear / activation / loss operations
        |
        v
pybind11 dispatch
        |
        v
C++ Tensor operations
        |
        v
CUDA forward kernels
        |
        v
Loss tensor
        |
        v
loss.backward()
        |
        v
Topological graph traversal
        |
        v
CUDA backward kernels
        |
        v
Parameter gradients
        |
        v
SGD or Adam step
        |
        v
Updated model parameters
```

---

## 18. Example: Linear Layer Forward Pass

Python:

```python
layer = mini.nn.Linear(3, 2)
x = mini.Tensor([[1.0, 2.0, 3.0]])

output = layer(x)
```

The Python layer executes:

```python
output = x @ layer.W + layer.b
```

Internal flow:

```text
x @ W
  |
  v
Tensor::matmul()
  |
  v
launch_matmul()
  |
  v
CUDA tiled matmul kernel
  |
  v
intermediate output
  |
  v
intermediate + bias
  |
  v
broadcast shape calculation
  |
  v
CUDA broadcast-add kernel
  |
  v
final output
```

The resulting autograd graph is:

```text
x -----\
        MATMUL ----\
W -----/            ADD ---> output
                   /
b ----------------/
```

---

## 19. Current Limitations

MiniPyPy is still experimental.

Current limitations include:

* Only float tensors are supported
* No integer tensor dtype
* No convolution layer
* No `no_grad()` context
* No in-place parameter update system
* Optimizer state is managed in Python
* Some graph-saving and cloning paths use CPU round trips
* Cross-entropy currently expects 2D logits
* Some reductions and helper operations are not fully generalized
* CUDA error handling is not yet centralized
* The memory pool does not yet support block splitting or stream awareness
* TensorFold layers are not yet implemented
* API and internals may change between versions

---

## 20. Planned Architecture Improvements

Future architectural work includes:

1. Implement `TensorFoldLinear`.
2. Add direct GPU-to-GPU cloning.
3. Add in-place parameter updates.
4. Introduce a `no_grad()` mechanism.
5. Improve CUDA error checking.
6. Reduce temporary metadata allocations.
7. Add more activation and mathematical operations.
8. Add convolution support.
9. Add model serialization.
10. Benchmark custom kernels against optimized CUDA libraries.
11. Investigate optional cuBLAS and cuDNN integration.
12. Improve the memory pool with stream-aware reuse.
13. Add dtype support.
14. Improve graph ownership and lifetime management.

---

## 21. TensorFold Integration

TensorFold will be implemented as a subsystem of MiniPyPy.

The first planned layer is:

```python
mini.nn.TensorFoldLinear(
    in_features,
    out_features,
    rank,
)
```

Instead of storing one dense weight matrix:

```text
W shape = [in_features, out_features]
```

the first implementation will store two smaller factors:

```text
U shape = [in_features, rank]
V shape = [rank, out_features]
```

Forward computation:

```text
output = (input @ U) @ V + bias
```

The full matrix:

```text
W = U @ V
```

will not be reconstructed during normal execution.

This design allows TensorFold to reuse MiniPyPy's existing:

* Tensor class
* Matrix multiplication
* Broadcasting
* Autograd system
* Neural-network module API
* Optimizers
* CUDA memory pool

Further TensorFold design details are documented in:

```text
future/tensorfold.md
```

---

## 22. Project Philosophy

MiniPyPy prioritizes:

```text
understanding before abstraction
correctness before optimization
measurement before performance claims
small components before broad APIs
```

Every major framework feature should be implemented with:

* A mathematical definition
* A clear ownership model
* A CPU-verifiable test
* A CUDA implementation
* A backward rule
* A benchmark where performance matters
* Documentation explaining the design decision

```
```
