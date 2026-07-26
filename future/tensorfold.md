# TensorFold

TensorFold is MiniPyPy's neural-network compression subsystem.

It is not a separate framework. It uses MiniPyPy's tensor engine, CUDA operations, autograd system, neural-network API, optimizers, and memory pool to provide factorized replacements for dense neural-network layers.

The first TensorFold component, `TensorFoldLinear`, is implemented.

`TensorFoldLinear` is currently a low-rank matrix factorization layer. It is not yet CP decomposition, Tucker decomposition, Tensor Train, TT-SVD, or HOSVD.

---

## 1. Current Implementation Status

Implemented in `v0.9.0`:

```text
mini.nn.TensorFoldLinear
```

Current TensorFold path:

```text
Train compressed low-rank neural networks from scratch.
```

Implemented features:

* Trainable low-rank factors
* Xavier initialization by default
* Optional simple initialization
* Parameter-count reporting
* Dense-parameter-count reporting
* Compression-ratio reporting
* Forward pass using existing MiniPyPy matmul
* Autograd through both factor matrices
* SGD and Adam support through MiniPyPy optimizers
* Single-layer MNIST benchmark
* Dense MLP vs TensorFold MLP benchmark
* Unit tests for construction, shapes, parameters, initialization, validation, and training behavior

Not implemented yet:

* CP decomposition
* Tucker decomposition
* Tensor Train decomposition
* TT-SVD
* HOSVD
* Dense-to-TensorFold conversion
* Pretrained PyTorch model compression
* Tensorized convolution layers
* Tensor Train CUDA kernels

---

## 2. Motivation

A normal dense linear layer stores a weight matrix:

```text
W ∈ R^(in_features × out_features)
```

and computes:

```text
Y = XW + b
```

The number of weight parameters is:

```text
dense_parameters = in_features × out_features
```

For large neural networks, dense layers can consume substantial:

* Parameter memory
* GPU memory
* Storage space
* Memory bandwidth
* Matrix-multiplication compute

TensorFold aims to replace selected dense layers with smaller factorized representations.

The main idea is:

```text
Do not store the full weight matrix.
Store smaller factors.
Apply the factors directly during forward execution.
```

---

## 3. TensorFold Is a MiniPyPy Subsystem

TensorFold lives inside MiniPyPy's neural-network API.

Current usage:

```python
import minipypy as mini

layer = mini.nn.TensorFoldLinear(
    in_features=784,
    out_features=256,
    rank=16,
)
```

A model can combine standard and TensorFold layers:

```python
model = mini.nn.Sequential(
    mini.nn.TensorFoldLinear(784, 256, rank=16),
    mini.nn.ReLU(),
    mini.nn.Linear(256, 10),
)
```

TensorFold reuses:

```text
MiniPyPy Tensor
MiniPyPy matmul
MiniPyPy broadcasting
MiniPyPy autograd
MiniPyPy Module
MiniPyPy SGD and Adam
MiniPyPy CUDA memory pool
```

No new C++ or CUDA operation is required for the current low-rank prototype.

---

## 4. Current Scope

The current TensorFold implementation focuses on two-dimensional linear-layer weight matrices.

For a normal linear layer:

```text
W ∈ R^(m × n)
```

TensorFold represents the effective weight as:

```text
W ≈ UV
```

where:

```text
U ∈ R^(m × r)
V ∈ R^(r × n)
```

and:

```text
r << min(m, n)
```

The value `r` is called the rank or chosen low-rank dimension.

This is low-rank matrix factorization.

Higher-order tensor decompositions such as Tucker, CP, and Tensor Train are planned future work.

---

## 5. Dense Linear Layer

Suppose:

```text
X ∈ R^(B × m)
W ∈ R^(m × n)
b ∈ R^(1 × n)
```

where:

```text
B = batch size
m = input features
n = output features
```

A dense layer computes:

```text
Y = XW + b
```

### Parameter count

Ignoring bias:

```text
P_dense = mn
```

Including bias:

```text
P_dense = mn + n
```

### Approximate multiply-add cost

The main matrix multiplication costs approximately:

```text
C_dense = Bmn
```

scalar multiply-accumulate operations.

---

## 6. TensorFoldLinear Layer

TensorFoldLinear replaces the full matrix with two factors:

```text
W ≈ UV
```

with:

```text
U ∈ R^(m × r)
V ∈ R^(r × n)
```

The forward pass becomes:

```text
H = XU
Y = HV + b
```

or:

```text
Y = (XU)V + b
```

The full matrix `W` does not need to be constructed.

### Shapes

```text
X: [B, m]
U: [m, r]
H: [B, r]
V: [r, n]
Y: [B, n]
```

### Parameter count

Ignoring bias:

```text
P_tensorfold = mr + rn
```

Including bias:

```text
P_tensorfold = mr + rn + n
```

### Approximate compute cost

```text
C_tensorfold = Bmr + Brn
```

or:

```text
C_tensorfold = Br(m + n)
```

---

## 7. Compression Condition

TensorFoldLinear uses fewer weight parameters when:

```text
mr + rn < mn
```

Factor out `r`:

```text
r(m + n) < mn
```

Therefore:

```text
r < mn / (m + n)
```

This value gives the maximum rank that still provides parameter compression.

For a square matrix where:

```text
m = n = d
```

the condition becomes:

```text
r < d / 2
```

However, meaningful compression normally requires a rank substantially smaller than this limit.

---

## 8. Compression Ratio

The weight compression ratio can be defined as:

```text
compression_ratio =
dense_weight_parameters / tensorfold_weight_parameters
```

Therefore:

```text
compression_ratio =
mn / (mr + rn)
```

or:

```text
compression_ratio =
mn / [r(m + n)]
```

A value greater than `1` means the factorized representation uses fewer parameters.

Example:

```text
m = 784
n = 256
r = 16
```

Dense parameters:

```text
784 × 256 = 200,704
```

TensorFold parameters:

```text
784 × 16 + 16 × 256
= 12,544 + 4,096
= 16,640
```

Compression ratio:

```text
200,704 / 16,640 ≈ 12.06
```

Bias parameters remain unchanged.

---

## 9. Compute Reduction

The compute ratio can be approximated as:

```text
dense_compute / tensorfold_compute
=
Bmn / [Br(m + n)]
```

The batch size cancels:

```text
compute_ratio =
mn / [r(m + n)]
```

This has the same algebraic form as the parameter compression ratio.

However, lower theoretical operation count does not automatically guarantee lower runtime.

Actual performance also depends on:

* GPU kernel-launch overhead
* Matrix dimensions
* Batch size
* Memory access
* CUDA occupancy
* Temporary tensor allocation
* Matmul-kernel efficiency
* Whether two smaller matmuls outperform one larger matmul

TensorFold performance must therefore be benchmarked rather than assumed.

---

## 10. Example Forward Pass

Consider:

```text
X shape = [32, 784]
U shape = [784, 16]
V shape = [16, 256]
b shape = [1, 256]
```

First multiplication:

```text
H = X @ U
```

Output shape:

```text
H shape = [32, 16]
```

Second multiplication:

```text
Y = H @ V
```

Output shape:

```text
Y shape = [32, 256]
```

Bias addition:

```text
Y = Y + b
```

Final output shape:

```text
[32, 256]
```

The equivalent full weight matrix would have shape:

```text
[784, 256]
```

but it is never reconstructed during normal inference or training.

---

## 11. Rank

The selected rank `r` determines the trade-off between compression and model capacity.

A smaller rank gives:

* Fewer parameters
* Lower theoretical compute
* Lower memory usage
* Stronger compression
* Greater risk of accuracy loss

A larger rank gives:

* More parameters
* More compute
* More expressive capacity
* Lower compression
* Better chance of preserving dense-layer accuracy

Rank is therefore a model-design hyperparameter.

It should not be chosen only from matrix dimensions. It must also be validated experimentally.

---

## 12. Rank Validation

The current implementation validates:

```text
in_features > 0
out_features > 0
rank > 0
init ∈ {"xavier", "simple"}
```

It also warns when the selected rank may not provide parameter compression.

The useful compression threshold is:

```text
rank < (in_features × out_features) / (in_features + out_features)
```

For example, for:

```text
in_features = 784
out_features = 10
```

the maximum useful compression rank is approximately:

```text
9.87
```

So:

```text
rank = 8   → compressed
rank = 10  → not compressed
```

---

## 13. Initialization

Initialization is important because the effective weight is:

```text
W_effective = UV
```

If both `U` and `V` are initialized poorly, the effective weight scale can become too small or too large.

The current implementation supports:

```python
mini.nn.TensorFoldLinear(784, 10, rank=4, init="xavier")
```

and:

```python
mini.nn.TensorFoldLinear(784, 10, rank=4, init="simple")
```

`xavier` is the default.

### Xavier initialization

The current implementation uses separate Xavier-style scales for `U` and `V`:

```text
U scale = sqrt(6 / (in_features + rank))
V scale = sqrt(6 / (rank + out_features))
```

This produced better MNIST behavior than the earlier very small fixed-scale initialization.

### Simple initialization

The simple initializer uses a fixed small scale:

```text
U scale = 0.05
V scale = 0.05
```

This is kept mainly for comparison and debugging.

---

## 14. Two TensorFold Workflows

TensorFold should eventually support two different workflows.

### Workflow A: Train Factorized Layer From Scratch

This workflow is implemented.

Initialize `U` and `V` directly and train them as model parameters.

```text
random U
random V
    |
    v
factorized forward pass
    |
    v
backpropagation
    |
    v
update U and V
```

Advantages:

* No pretrained dense model required
* Full training occurs in compressed form
* Full weight matrix is never stored

Challenges:

* Optimization may behave differently from dense training
* Initialization requires care
* Low rank may restrict learning capacity

### Workflow B: Compress a Pretrained Dense Layer

This workflow is future work.

Start with a trained dense weight matrix:

```text
W
```

Apply truncated SVD:

```text
W ≈ U_r Σ_r V_rᵀ
```

Convert this into two factors.

One valid split is:

```text
A = U_r Σ_r
B = V_rᵀ
```

so:

```text
W ≈ AB
```

Another balanced split is:

```text
A = U_r Σ_r^(1/2)
B = Σ_r^(1/2)V_rᵀ
```

so again:

```text
W ≈ AB
```

The factors can then be:

* Used directly for inference
* Fine-tuned using MiniPyPy autograd
* Compared against the original dense layer

---

## 15. SVD Mathematics for Future Dense-to-TensorFold Conversion

For a matrix:

```text
W ∈ R^(m × n)
```

the singular value decomposition is:

```text
W = UΣVᵀ
```

where:

```text
U ∈ R^(m × m)
Σ ∈ R^(m × n)
V ∈ R^(n × n)
```

In reduced form:

```text
U ∈ R^(m × k)
Σ ∈ R^(k × k)
V ∈ R^(n × k)
```

where:

```text
k = min(m, n)
```

A rank-`r` approximation keeps only the first `r` singular components:

```text
W_r = U_r Σ_r V_rᵀ
```

where:

```text
U_r ∈ R^(m × r)
Σ_r ∈ R^(r × r)
V_rᵀ ∈ R^(r × n)
```

The truncated SVD gives the best rank-`r` approximation under the Frobenius norm and spectral norm.

TensorFold can store:

```text
A = U_r Σ_r
B = V_rᵀ
```

with:

```text
A shape = [m, r]
B shape = [r, n]
```

Then:

```text
W_r = AB
```

This is planned for a later milestone.

---

## 16. Reconstruction Error

For a dense weight matrix `W` and its approximation `W_r`, the Frobenius reconstruction error is:

```text
error = ||W - W_r||_F
```

A relative error can be defined as:

```text
relative_error =
||W - W_r||_F / ||W||_F
```

Using singular values, the squared Frobenius error of truncated SVD is:

```text
||W - W_r||_F² =
Σ from i=r+1 to k of σᵢ²
```

This gives a mathematical measure of how much information is removed by the selected rank.

However, low reconstruction error does not always guarantee unchanged neural-network accuracy.

The layer must also be evaluated within the full model.

---

## 17. Current TensorFoldLinear API

Current API:

```python
class TensorFoldLinear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        init="xavier",
    ):
        ...
```

Usage:

```python
layer = mini.nn.TensorFoldLinear(
    in_features=784,
    out_features=256,
    rank=16,
)
```

Current attributes:

```python
layer.in_features
layer.out_features
layer.rank
layer.init

layer.U
layer.V
layer.b
```

Forward pass:

```python
def forward(self, x):
    hidden = x @ self.U
    out = hidden @ self.V
    return out + self.b
```

Parameter collection:

```python
def parameters(self):
    return [self.U, self.V, self.b]
```

Named parameter collection:

```python
def named_parameters(self):
    return [
        (self, "U", self.U),
        (self, "V", self.V),
        (self, "b", self.b),
    ]
```

Parameter count:

```python
def parameter_count(self):
    return (
        self.in_features * self.rank
        + self.rank * self.out_features
        + self.out_features
    )
```

Dense parameter count:

```python
def dense_parameter_count(self):
    return (
        self.in_features * self.out_features
        + self.out_features
    )
```

Compression ratio:

```python
def compression_ratio(self):
    return self.dense_parameter_count() / self.parameter_count()
```

---

## 18. Autograd

No new autograd operation is required for the current prototype.

The forward pass consists of existing differentiable operations:

```text
H = X @ U
Y = H @ V
Y = Y + b
```

MiniPyPy already supports backward rules for:

* Matmul
* Addition
* Broadcasting

The autograd graph looks like:

```text
X -----\
        MATMUL ---> H -----\
U -----/                    MATMUL ----\
                           /            ADD ---> Y
V ------------------------/            /
                                      b
```

Gradients are:

```text
dL/dV = Hᵀ @ dL/dY
```

```text
dL/dH = dL/dY @ Vᵀ
```

```text
dL/dU = Xᵀ @ dL/dH
```

```text
dL/dX = dL/dH @ Uᵀ
```

Bias gradient is reduced over broadcast dimensions.

---

## 19. Why the Full Weight Matrix Is Not Reconstructed

A naive implementation could compute:

```text
W = U @ V
Y = X @ W
```

This would reconstruct the full dense matrix.

That approach loses important TensorFold benefits because it:

* Allocates the original large matrix
* Adds an extra matrix multiplication
* Increases temporary memory
* Reduces inference efficiency
* Defeats the purpose of storing only factors

TensorFoldLinear instead computes:

```text
Y = (X @ U) @ V
```

This directly contracts the factors with the input.

---

## 20. Intermediate Activation Cost

TensorFoldLinear introduces an intermediate tensor:

```text
H = X @ U
```

with shape:

```text
[B, r]
```

This intermediate is usually smaller than the dense output when:

```text
r < out_features
```

Memory for the intermediate is:

```text
B × r
```

This should be included in GPU-memory benchmarks.

During training, autograd may retain additional values required for backward.

Therefore benchmarks must distinguish between:

* Parameter memory
* Forward activation memory
* Training graph memory
* Peak allocated memory
* Cached memory-pool memory

---

## 21. When TensorFold May Not Help

TensorFold will not automatically improve every layer.

It may be ineffective when:

* The layer is already small
* The selected rank is too large
* The weight matrix is not approximately low rank
* Kernel-launch overhead dominates execution
* Batch sizes are very small
* The two factorized matmuls are poorly shaped for the GPU
* Accuracy loss requires a high rank
* The layer contributes little to total model size
* Another model component is the actual bottleneck

TensorFold should therefore replace layers selectively.

---

## 22. Candidate Layers for Compression

Good initial candidates include:

* Large fully connected layers
* MLP hidden layers
* Transformer projection matrices
* Transformer feed-forward layers
* Large classification heads
* Dense layers used in edge inference models

Poor initial candidates include:

* Very small output classifiers
* Layers whose parameters are a negligible part of the model
* Layers where low-rank approximation causes severe accuracy loss

The current MiniPyPy benchmark remains small and understandable, even though the absolute runtime improvement may be limited.

---

## 23. Completed Benchmarks

### Single-layer MNIST benchmark

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

Observed result summary:

```text
rank=8 nearly matched the dense single-layer classifier while using fewer parameters.
rank=10 reached slightly higher accuracy but was not a compression win.
```

### Dense MLP vs TensorFold MLP benchmark

Dense MLP:

```python
model = mini.nn.Sequential(
    mini.nn.Linear(784, 128),
    mini.nn.ReLU(),
    mini.nn.Linear(128, 10),
)
```

TensorFold variants replace one or both dense layers with `TensorFoldLinear`.

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

Key observed result:

```text
TensorFold MLP r32/r10 reached dense-level accuracy while using about 3.31x fewer parameters.
```

Important interpretation:

```text
Compressing every layer is not always optimal.
Compressing large dense layers gives the best parameter savings.
Small final classifier layers may sometimes be better left dense or given higher rank.
```

---

## 24. Required Tests

Implemented tests currently verify:

* Forward output shape
* Parameter list
* Named parameter list
* Parameter count
* Training reduces CrossEntropyLoss
* Simple initialization works
* Xavier initialization works
* Unknown initialization is rejected
* Invalid `in_features` is rejected
* Invalid `out_features` is rejected
* Default initialization is Xavier

Additional future TensorFold tests should include:

### Dense Equivalence Test

Set:

```text
W = U @ V
```

Compare:

```text
TensorFold output = (X @ U) @ V + b
Dense output = X @ W + b
```

The values should agree within floating-point tolerance.

### Gradient Tests

Verify gradients exist for:

```text
U
V
bias
input
```

Compare selected small cases against finite differences or a trusted reference implementation.

### Compression Test

Verify the reported compression ratio matches the mathematical formula across multiple layer sizes and ranks.

---

## 25. Benchmark Methodology

CUDA operations are asynchronous.

Timing code must synchronize before reading elapsed wall-clock time.

Otherwise, the measured duration may include only kernel launch time instead of actual GPU execution.

Benchmarks should also separate:

* First-run initialization overhead
* Warmed-up execution
* Allocation overhead
* Data transfer
* Forward execution
* Backward execution
* Optimizer update

A possible process is:

```text
1. Build model.
2. Run warm-up iterations.
3. Synchronize CUDA.
4. Start timer.
5. Run measured iterations.
6. Synchronize CUDA.
7. Stop timer.
8. Report median and distribution.
```

TensorFold claims should be based on repeatable measurements rather than a single run.

---

## 26. Completed and Future Implementation Phases

### Phase 1: Mathematics and API

Status: complete for low-rank TensorFoldLinear.

Completed:

* Factor orientation
* Rank validation
* Parameter-count formulas
* Compression-ratio reporting
* Initialization design
* Documentation

### Phase 2: Python Prototype

Status: complete.

Completed:

* Implemented `TensorFoldLinear` using existing matmul
* Added it to `mini.nn`
* Added constructor and forward tests
* Added training test
* Added initialization tests
* Added validation tests

### Phase 3: Dense Comparison

Status: partially complete.

Completed:

* Compared parameter counts
* Compared dense MLP against TensorFold MLP variants
* Benchmarked several ranks

Future:

* Add dense-equivalence output test where `W = U @ V`
* Add more gradient reference tests

### Phase 4: MNIST Experiment

Status: complete for first prototype.

Completed:

* Trained dense and factorized models
* Evaluated several ranks
* Recorded loss and accuracy
* Recorded parameter count
* Compared against dense baseline

Future:

* Add timing measurements
* Add peak GPU memory measurements
* Add repeated-seed benchmark summaries

### Phase 5: Pretrained Compression

Status: future work.

Planned:

* Add an SVD conversion workflow
* Convert a trained dense layer
* Measure reconstruction error
* Fine-tune compressed factors
* Compare pre- and post-compression accuracy

### Phase 6: CUDA Optimization

Status: future work.

Planned:

* Profile both matmuls
* Inspect memory-pool behavior
* Evaluate fused execution possibilities
* Reduce temporary allocation
* Consider specialized kernels only after profiling

### Phase 7: Higher-Order TensorFold

Status: future work.

Planned:

* Investigate tensorization of weight matrices
* Study Tucker decomposition
* Study CP decomposition
* Study Tensor Train
* Select one decomposition based on measurable use cases

---

## 27. Potential Fused CUDA Kernel

The current TensorFold implementation uses two ordinary matmuls:

```text
H = X @ U
Y = H @ V
```

A future fused kernel may attempt to reduce the temporary storage of `H`.

Conceptually:

```text
for each output element Y[b, n]:
    sum over rank r:
        V[r, n] ×
        sum over input m:
            X[b, m] × U[m, r]
```

However, direct fusion may:

* Recompute intermediate values
* Increase register pressure
* Complicate tiling
* Reduce reuse
* Perform worse than two optimized matmuls

A fused kernel should only be implemented after profiling demonstrates a clear benefit.

---

## 28. Higher-Order Tensor Decompositions

Low-rank matrix factorization is appropriate for a two-dimensional matrix.

Future TensorFold versions may tensorize a large matrix into multiple modes.

For example:

```text
W shape = [m, n]
```

could be reshaped conceptually into:

```text
W_tensor shape =
[m1, m2, ..., md, n1, n2, ..., nd]
```

where:

```text
m = m1m2...md
n = n1n2...nd
```

Possible decompositions include:

### CP Decomposition

Represents a tensor as a sum of rank-one outer products.

### Tucker Decomposition

Represents a tensor using a smaller core tensor multiplied by factor matrices.

### Tensor Train

Represents a high-order tensor as a sequence of small three-dimensional cores.

These approaches may provide stronger compression than ordinary matrix rank factorization, but they introduce more complex:

* Shape selection
* Rank selection
* Contraction order
* Kernel design
* Autograd paths
* Initialization
* Conversion algorithms

They should follow, not precede, a successful `TensorFoldLinear` prototype.

---

## 29. Tensor Train Direction

A Tensor Train representation uses cores:

```text
G₁, G₂, ..., G_d
```

with shapes similar to:

```text
G_k ∈ R^(r_{k-1} × n_k × r_k)
```

where:

```text
r_0 = 1
r_d = 1
```

The values `r_k` are Tensor Train ranks.

A Tensor Train layer could significantly reduce parameters when a large dimension can be factorized into several smaller modes.

However, TT implementation requires:

* Tensorization rules
* TT-rank selection
* Efficient contraction order
* TT-SVD for pretrained conversion
* Specialized tests
* Careful GPU-kernel design

Tensor Train is a later TensorFold milestone.

---

## 30. Research Questions

TensorFold should investigate the following questions experimentally:

1. Which MiniPyPy layer sizes benefit from factorization?
2. At what ranks does accuracy begin to degrade?
3. Does training from scratch behave differently from SVD compression?
4. How much fine-tuning is required after compression?
5. When do two small matmuls outperform one dense matmul?
6. How does batch size affect performance?
7. How much peak GPU memory is saved?
8. Which rank-selection strategy works best?
9. Which layers should remain dense?
10. Can factorization improve edge-device deployment?
11. When is SVD sufficient?
12. When are Tucker or Tensor Train representations worthwhile?
13. Can TensorFold layers be exported independently of MiniPyPy?
14. Can an existing model be converted automatically?
15. Can decomposed factors be quantized after compression?

---

## 31. Success Criteria

The first TensorFold milestone is successful when:

```text
TensorFoldLinear exists inside mini.nn
```

and:

* It stores factor matrices instead of a full weight matrix
* Forward execution does not reconstruct the full matrix
* Autograd computes gradients through the factorized path
* SGD and Adam can train the factors
* Unit tests pass
* Parameter-count reduction is reported
* An MNIST model trains successfully
* Accuracy is compared against a dense baseline
* Limitations are documented

For the current `v0.9.0` stage, this milestone is substantially complete.

Remaining improvements:

* Add dense-equivalence test
* Add explicit gradient-presence tests for U, V, bias, and input
* Add timing benchmarks
* Add memory benchmarks
* Add repeated-seed benchmark runs

---

## 32. Non-Goals for the First TensorFold Version

The first TensorFold release does not attempt to provide:

* Automatic compression of every model type
* Convolution decomposition
* Transformer-wide conversion
* Tensor Train CUDA kernels
* Production deployment tooling
* Distributed training
* Mixed precision
* Quantization
* Universal inference speedup
* Guaranteed accuracy preservation
* Automatic optimal-rank discovery

These features may be explored after the basic layer is stable.

---

## 33. Proposed Roadmap

```text
v0.9.0
TensorFoldLinear low-rank prototype

v0.9.1
TensorFoldLinear documentation, tests, and packaging cleanup

v0.10.0
True tensorized / Tensor Train Linear prototype

v0.11.0
Pretrained dense-to-TensorFold compression

v0.12.0
Rank-selection utilities and compression reports

Later
CUDA optimization and higher-order decompositions
```

The exact version numbers may change as MiniPyPy evolves.

---

## 34. Core Design Principle

TensorFold's defining principle is:

```text
Store factors only.
Operate on factors directly.
Never reconstruct the full dense weight during normal execution.
```

The current computational path is:

```text
Input X
   |
   v
X @ U
   |
   v
Low-dimensional representation H
   |
   v
H @ V
   |
   v
Add bias
   |
   v
Output Y
```

This keeps TensorFold aligned with MiniPyPy's larger goal:

```text
A small, understandable, GPU-backed framework
with native support for compressed neural-network layers.
```
