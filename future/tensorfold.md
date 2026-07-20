# TensorFold

TensorFold is MiniPyPy's neural-network compression subsystem.

It is not a separate framework. It will use MiniPyPy's tensor engine, CUDA operations, autograd system, neural-network API, optimizers, and memory pool to provide factorized replacements for dense neural-network layers.

The first TensorFold component will be `TensorFoldLinear`.

---

## 1. Motivation

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

## 2. TensorFold Is a MiniPyPy Subsystem

TensorFold will live inside MiniPyPy's neural-network API.

Planned usage:

```python
import minipypy as mini

layer = mini.nn.TensorFoldLinear(
    in_features=784,
    out_features=256,
    rank=16,
)
```

A model may combine standard and TensorFold layers:

```python
model = mini.nn.Sequential(
    mini.nn.TensorFoldLinear(784, 256, rank=16),
    mini.nn.ReLU(),
    mini.nn.Linear(256, 10),
)
```

TensorFold will reuse:

```text
MiniPyPy Tensor
MiniPyPy matmul
MiniPyPy broadcasting
MiniPyPy autograd
MiniPyPy Module
MiniPyPy SGD and Adam
MiniPyPy CUDA memory pool
```

---

## 3. Initial Scope

The first TensorFold implementation will focus on two-dimensional linear-layer weight matrices.

For a normal linear layer:

```text
W ∈ R^(m × n)
```

TensorFold will approximate or represent it as:

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

The first implementation is therefore a low-rank matrix factorization layer.

Higher-order tensor decompositions such as Tucker, CP, or Tensor Train may be explored after the low-rank linear layer is correct and benchmarked.

---

## 4. Dense Linear Layer

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

## 5. TensorFold Linear Layer

TensorFold replaces the full matrix with two factors:

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

## 6. Compression Condition

TensorFold uses fewer weight parameters when:

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

## 7. Compression Ratio

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
mn / (r(m + n))
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

The factorized weights use approximately twelve times fewer parameters.

Bias parameters remain unchanged.

---

## 8. Compute Reduction

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

## 9. Example Forward Pass

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

## 10. Rank

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

## 11. Rank Selection Strategies

TensorFold may support several rank-selection strategies.

### 11.1 Fixed Rank

The user explicitly specifies the rank:

```python
mini.nn.TensorFoldLinear(784, 256, rank=16)
```

This is the simplest first implementation.

### 11.2 Compression-Target Rank

Choose a rank based on a desired compression ratio `c`.

Starting with:

```text
c = mn / [r(m + n)]
```

solve for `r`:

```text
r = mn / [c(m + n)]
```

The result must be rounded to a valid positive integer.

Example:

```text
m = 784
n = 256
desired compression = 8
```

Then:

```text
r = (784 × 256) / [8 × (784 + 256)]
```

```text
r ≈ 24.12
```

A practical choice could be:

```text
r = 24
```

### 11.3 Energy-Based Rank From SVD

For a pretrained dense matrix:

```text
W = UΣVᵀ
```

the singular values are:

```text
σ₁ ≥ σ₂ ≥ ... ≥ σ_k
```

An energy-retention threshold can be used:

```text
retained_energy(r) =
(Σ from i=1 to r of σᵢ²)
/
(Σ from i=1 to k of σᵢ²)
```

Choose the smallest rank `r` such that:

```text
retained_energy(r) ≥ threshold
```

Possible thresholds include:

```text
0.90
0.95
0.99
```

This strategy is useful when compressing an already-trained model.

### 11.4 Validation-Based Rank

Train or evaluate multiple ranks:

```text
r ∈ {4, 8, 16, 32, 64}
```

Then compare:

* Validation loss
* Accuracy
* Parameter count
* GPU memory
* Training time
* Inference time

The best rank is selected according to the target deployment constraints.

---

## 12. Two TensorFold Workflows

TensorFold should eventually support two different workflows.

### Workflow A: Train Factorized Layer From Scratch

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

## 13. SVD Mathematics

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

---

## 14. Reconstruction Error

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

## 15. TensorFoldLinear API

The initial planned API is:

```python
class TensorFoldLinear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
    ):
        ...
```

Expected usage:

```python
layer = mini.nn.TensorFoldLinear(
    in_features=784,
    out_features=256,
    rank=16,
)
```

Potential attributes:

```python
layer.in_features
layer.out_features
layer.rank

layer.U
layer.V
layer.b
```

Forward pass:

```python
def forward(self, x):
    out = (x @ self.U) @ self.V

    if self.b is not None:
        out = out + self.b

    return out
```

Parameter collection:

```python
def parameters(self):
    params = [self.U, self.V]

    if self.b is not None:
        params.append(self.b)

    return params
```

---

## 16. Initial Python Prototype

The first version can be implemented entirely using existing MiniPyPy operations.

```python
class TensorFoldLinear(Module):
    def __init__(
        self,
        in_features,
        out_features,
        rank,
        bias=True,
    ):
        if rank <= 0:
            raise ValueError("rank must be positive")

        max_rank = min(in_features, out_features)

        if rank > max_rank:
            raise ValueError(
                "rank cannot exceed min(in_features, out_features)"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = make_parameter(in_features, rank)
        self.V = make_parameter(rank, out_features)

        self.b = (
            make_zero_parameter(1, out_features)
            if bias
            else None
        )

    def forward(self, x):
        hidden = x @ self.U
        output = hidden @ self.V

        if self.b is not None:
            output = output + self.b

        return output
```

The exact initialization helpers will depend on MiniPyPy's module utilities.

The prototype should initially prioritize correctness over custom fused kernels.

---

## 17. Autograd

No new autograd operation is required for the first prototype.

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

The autograd graph will look like:

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

## 18. Initialization

Initialization is important because the effective weight is:

```text
W_effective = UV
```

If both `U` and `V` are initialized with very small values, their product may become excessively small.

Possible initial strategies include:

### Strategy A: Scale-Aware Random Factors

Initialize both factors with variance chosen so that the product has a reasonable scale.

A simple first approach may use:

```text
U scale ≈ 1 / sqrt(in_features)
V scale ≈ 1 / sqrt(rank)
```

This must be validated experimentally.

### Strategy B: Dense Initialization Followed by SVD

1. Generate a normal dense initialization.
2. Apply rank-`r` SVD.
3. Store the resulting factors.

This more closely matches the initialization statistics of a standard linear layer but requires an SVD implementation or external preprocessing.

### Strategy C: One Random Factor and One Structured Factor

One factor could be initialized randomly while the other starts close to an identity-like mapping where dimensions permit.

This is more specialized and should not be the first implementation.

The first prototype should use a simple documented initialization and verify that training loss decreases.

---

## 19. Dense-to-TensorFold Conversion

A future conversion utility could look like:

```python
compressed = mini.nn.TensorFoldLinear.from_linear(
    dense_layer,
    rank=16,
)
```

Conceptual conversion:

```text
dense W
   |
   v
SVD(W)
   |
   v
truncate to rank r
   |
   v
construct factor U
   |
   v
construct factor V
   |
   v
copy dense bias
   |
   v
TensorFoldLinear
```

A model-level utility could recursively replace selected layers:

```python
compressed_model = mini.tensorfold.compress(
    model,
    rank=16,
)
```

A more advanced API may allow per-layer rank configuration:

```python
compressed_model = mini.tensorfold.compress(
    model,
    ranks={
        "layer1": 32,
        "layer2": 16,
        "classifier": 8,
    },
)
```

These APIs are future work and should not be implemented before the core layer is stable.

---

## 20. Why the Full Weight Matrix Should Not Be Reconstructed

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

TensorFold should instead compute:

```text
Y = (X @ U) @ V
```

This directly contracts the factors with the input.

---

## 21. Intermediate Activation Cost

TensorFold introduces an intermediate tensor:

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

## 22. When TensorFold May Not Help

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

## 23. Candidate Layers for Compression

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

The first MiniPyPy benchmark should remain small and understandable, even if the absolute runtime improvement is limited.

---

## 24. First Benchmark

The first controlled benchmark should compare:

```text
Dense:
mini.nn.Linear(784, 256)

TensorFold:
mini.nn.TensorFoldLinear(784, 256, rank=16)
```

Both models should use:

* The same dataset
* The same random seed where possible
* The same number of epochs
* The same batch size
* The same optimizer
* Comparable initialization
* The same loss function
* The same evaluation set

Metrics should include:

```text
Parameter count
Compression ratio
Training loss
Validation loss
Training accuracy
Validation accuracy
Forward latency
Training-step latency
Peak GPU memory
Final model-storage size
```

---

## 25. MNIST Prototype

An initial model could be:

```python
model = mini.nn.Sequential(
    mini.nn.TensorFoldLinear(784, 128, rank=16),
    mini.nn.ReLU(),
    mini.nn.Linear(128, 10),
)
```

Dense baseline:

```python
model = mini.nn.Sequential(
    mini.nn.Linear(784, 128),
    mini.nn.ReLU(),
    mini.nn.Linear(128, 10),
)
```

The first goal is not to prove universal superiority.

The first goal is to verify that:

1. Forward shapes are correct.
2. Gradients reach both factors.
3. Loss decreases.
4. Optimizers update both factors.
5. The layer trains end to end.
6. Parameter count is reduced.
7. Accuracy and runtime can be measured reliably.

---

## 26. Required Tests

### Constructor Tests

Verify:

```text
U shape = [in_features, rank]
V shape = [rank, out_features]
bias shape = [1, out_features]
```

Reject:

```text
rank <= 0
rank > min(in_features, out_features)
in_features <= 0
out_features <= 0
```

### Forward Shape Test

```text
input:  [batch, in_features]
output: [batch, out_features]
```

### Numerical Equivalence Test

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

### Training Test

Verify that TensorFoldLinear reduces a simple regression or classification loss.

### Parameter Count Test

Verify:

```text
actual factor parameters =
in_features × rank
+ rank × out_features
+ bias parameters
```

### Compression Test

Verify the reported compression ratio matches the mathematical formula.

---

## 27. Benchmark Methodology

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

## 28. Implementation Phases

### Phase 1: Mathematics and API

* Finalize factor orientation
* Define rank validation
* Define parameter-count formulas
* Define compression-ratio reporting
* Document initialization

### Phase 2: Python Prototype

* Implement `TensorFoldLinear` using existing matmul
* Add it to `mini.nn`
* Add constructor and forward tests
* Add gradient tests
* Add training test

### Phase 3: Dense Comparison

* Create equivalent dense baseline
* Compare parameter counts
* Compare outputs when `W = UV`
* Compare training behavior

### Phase 4: MNIST Experiment

* Train dense and factorized models
* Evaluate several ranks
* Record loss and accuracy
* Record parameter count
* Record execution time

### Phase 5: Pretrained Compression

* Add an SVD conversion workflow
* Convert a trained dense layer
* Measure reconstruction error
* Fine-tune compressed factors
* Compare pre- and post-compression accuracy

### Phase 6: CUDA Optimization

* Profile both matmuls
* Inspect memory-pool behavior
* Evaluate fused execution possibilities
* Reduce temporary allocation
* Consider specialized kernels only after profiling

### Phase 7: Higher-Order TensorFold

* Investigate tensorization of weight matrices
* Study Tucker decomposition
* Study CP decomposition
* Study Tensor Train
* Select one decomposition based on measurable use cases

---

## 29. Potential Fused CUDA Kernel

The first TensorFold implementation should use two ordinary matmuls:

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

## 30. Higher-Order Tensor Decompositions

Low-rank SVD factorization is appropriate for a two-dimensional matrix.

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

## 31. Tensor Train Direction

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

## 32. Research Questions

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

## 33. Success Criteria

The first TensorFold milestone is successful when:

```text
TensorFoldLinear exists inside mini.nn
```

and:

* It stores factor matrices instead of a full weight matrix
* Forward execution does not reconstruct the full matrix
* Autograd computes correct gradients
* SGD and Adam can train the factors
* Unit tests pass
* A dense-equivalence test passes
* Parameter-count reduction is reported
* An MNIST model trains successfully
* Accuracy is compared against a dense baseline
* Runtime is benchmarked honestly
* Limitations are documented

---

## 34. Non-Goals for the First Version

The first TensorFold release will not attempt to provide:

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

## 35. Proposed Roadmap

```text
v0.9.0
TensorFoldLinear Python prototype

v0.9.x
Correctness, gradient, and training tests

v0.10.0
Dense vs TensorFold MNIST benchmark

v0.11.0
Pretrained dense-layer SVD conversion

v0.12.0
Rank-selection utilities and compression reports

Later
CUDA optimization and higher-order decompositions
```

The exact version numbers may change as MiniPyPy evolves.

---

## 36. Core Design Principle

TensorFold's defining principle is:

```text
Store factors only.
Operate on factors directly.
Never reconstruct the full dense weight during normal execution.
```

The initial computational path is:

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
