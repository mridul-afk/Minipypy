#include "tensor.h"
#include "memory_pool.h"
#include "autograd.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
#include <algorithm>

// CUDA launchers from tensor.cu
void launch_add(const float *a, const float *b, float *out, int size);
void launch_mul(const float *a, const float *b, float *out, int size);
void launch_sub(const float *a, const float *b, float *out, int size);
void launch_div(const float *a, const float *b, float *out, int size);

void launch_matmul(const float *a, const float *b, float *out, int M, int N, int K);

void launch_batched_matmul(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_size);

void launch_broadcasted_batched_matmul(
    const float *A,
    const float *B,
    float *C,
    const int *a_batch_shape,
    const int *a_batch_stride,
    const int *b_batch_shape,
    const int *b_batch_stride,
    const int *out_batch_shape,
    int a_batch_ndim,
    int b_batch_ndim,
    int out_batch_ndim,
    int M,
    int N,
    int K,
    int out_batch_size);

void launch_transpose(const float *input, float *output, int rows, int cols);
void launch_sum(const float *input, float *output, int size);
void launch_divide_scalar(float *data, float scalar);
void launch_max(const float *input, float *output, int size);

void launch_add_broadcast(
    const float *a,
    const float *b,
    float *out,
    const int *a_shape,
    const int *a_stride,
    const int *b_shape,
    const int *b_stride,
    const int *out_shape,
    int a_ndim,
    int b_ndim,
    int out_ndim,
    int out_size);

void launch_sub_broadcast(
    const float *a,
    const float *b,
    float *out,
    const int *a_shape,
    const int *a_stride,
    const int *b_shape,
    const int *b_stride,
    const int *out_shape,
    int a_ndim,
    int b_ndim,
    int out_ndim,
    int out_size);

void launch_mul_broadcast(
    const float *a,
    const float *b,
    float *out,
    const int *a_shape,
    const int *a_stride,
    const int *b_shape,
    const int *b_stride,
    const int *out_shape,
    int a_ndim,
    int b_ndim,
    int out_ndim,
    int out_size);

void launch_div_broadcast(
    const float *a,
    const float *b,
    float *out,
    const int *a_shape,
    const int *a_stride,
    const int *b_shape,
    const int *b_stride,
    const int *out_shape,
    int a_ndim,
    int b_ndim,
    int out_ndim,
    int out_size);

void launch_transpose_last_two_dims(
    const float *input,
    float *output,
    const int *in_shape,
    const int *out_shape,
    int ndim,
    int total_size);

void launch_sum_to_shape(
    const float *input,
    float *output,
    const int *in_shape,
    const int *out_shape,
    const int *out_stride,
    int in_ndim,
    int out_ndim,
    int input_size);

std::vector<int> broadcast_shape(
    const std::vector<int> &a,
    const std::vector<int> &b)
{
  int ndim_a = static_cast<int>(a.size());
  int ndim_b = static_cast<int>(b.size());

  int ndim_out = std::max(ndim_a, ndim_b);
  std::vector<int> out_shape(ndim_out);

  for (int i = 0; i < ndim_out; i++)
  {
    int dim_a = 1;
    int dim_b = 1;

    int index_a = ndim_a - 1 - i;
    int index_b = ndim_b - 1 - i;

    if (index_a >= 0)
      dim_a = a[index_a];

    if (index_b >= 0)
      dim_b = b[index_b];

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1)
      throw std::runtime_error("Shapes are not broadcastable");

    out_shape[ndim_out - 1 - i] = std::max(dim_a, dim_b);
  }

  return out_shape;
}

std::vector<int> batch_shape_of_matmul(const std::vector<int> &shape)
{
  if (shape.size() < 2)
    throw std::runtime_error("matmul requires tensors with at least 2 dimensions");

  return std::vector<int>(shape.begin(), shape.end() - 2);
}

std::vector<int> compute_stride_from_shape(const std::vector<int> &shape)
{
  std::vector<int> stride(shape.size());

  int running = 1;

  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
  {
    stride[i] = running;
    running *= shape[i];
  }

  return stride;
}

int numel_from_shape(const std::vector<int> &shape)
{
  int total = 1;

  for (int dim : shape)
    total *= dim;

  return total;
}

std::vector<int> matmul_output_shape(
    const std::vector<int> &a_shape,
    const std::vector<int> &b_shape)
{
  if (a_shape.size() < 2 || b_shape.size() < 2)
    throw std::runtime_error("matmul requires tensors with at least 2 dimensions");

  int M = a_shape[a_shape.size() - 2];
  int K = a_shape[a_shape.size() - 1];

  int K_other = b_shape[b_shape.size() - 2];
  int N = b_shape[b_shape.size() - 1];

  if (K != K_other)
    throw std::runtime_error("matmul shape mismatch");

  std::vector<int> a_batch = batch_shape_of_matmul(a_shape);
  std::vector<int> b_batch = batch_shape_of_matmul(b_shape);

  std::vector<int> out_batch = broadcast_shape(a_batch, b_batch);

  std::vector<int> out_shape = out_batch;
  out_shape.push_back(M);
  out_shape.push_back(N);

  return out_shape;
}

int *copy_int_vector_to_device(const std::vector<int> &host)
{
  size_t bytes = host.size() * sizeof(int);

  int *d_ptr = static_cast<int *>(
      CudaMemoryPool::instance().allocate(bytes));

  cudaMemcpy(
      d_ptr,
      host.data(),
      bytes,
      cudaMemcpyHostToDevice);

  return d_ptr;
}

void free_device_int_vector(int *ptr, const std::vector<int> &host)
{
  if (!ptr)
    return;

  size_t bytes = host.size() * sizeof(int);

  CudaMemoryPool::instance().deallocate(ptr, bytes);
}

Tensor::Tensor(int size, bool requires_grad)
{
  this->size = size;
  this->shape = {size};
  this->stride = {1};
  this->is_cuda = true;
  this->d_data = nullptr;
  this->requires_grad = requires_grad;
  this->grad_tensor = nullptr;
  this->grad_fn = nullptr;

  d_data = static_cast<float *>(
      CudaMemoryPool::instance().allocate(size * sizeof(float)));
}

Tensor::Tensor(const std::vector<float> &host_data, bool requires_grad)
{
  this->size = static_cast<int>(host_data.size());
  this->shape = {size};
  this->stride = {1};
  this->is_cuda = true;
  this->d_data = nullptr;
  this->requires_grad = requires_grad;
  this->grad_tensor = nullptr;
  this->grad_fn = nullptr;

  d_data = static_cast<float *>(
      CudaMemoryPool::instance().allocate(size * sizeof(float)));

  cudaMemcpy(
      d_data,
      host_data.data(),
      size * sizeof(float),
      cudaMemcpyHostToDevice);
}

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
{
  this->shape = shape;
  this->size = 1;
  this->is_cuda = true;
  this->d_data = nullptr;
  this->requires_grad = requires_grad;
  this->grad_tensor = nullptr;
  this->grad_fn = nullptr;

  for (int dim : shape)
    this->size *= dim;

  this->stride.resize(shape.size());

  int running_stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
  {
    stride[i] = running_stride;
    running_stride *= shape[i];
  }

  d_data = static_cast<float *>(
      CudaMemoryPool::instance().allocate(size * sizeof(float)));
}

Tensor::Tensor(
    const std::vector<float> &host_data,
    std::vector<int> shape,
    bool requires_grad)
{
  this->shape = shape;
  this->size = 1;
  this->is_cuda = true;
  this->d_data = nullptr;
  this->requires_grad = requires_grad;
  this->grad_tensor = nullptr;
  this->grad_fn = nullptr;

  for (int dim : shape)
    this->size *= dim;

  if (this->size != static_cast<int>(host_data.size()))
    throw std::runtime_error("Data size does not match tensor shape");

  this->stride.resize(shape.size());

  int running_stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
  {
    stride[i] = running_stride;
    running_stride *= shape[i];
  }

  d_data = static_cast<float *>(
      CudaMemoryPool::instance().allocate(size * sizeof(float)));

  cudaMemcpy(
      d_data,
      host_data.data(),
      size * sizeof(float),
      cudaMemcpyHostToDevice);
}

Tensor::Tensor(Tensor &&other) noexcept
{
  this->d_data = other.d_data;
  this->size = other.size;
  this->shape = std::move(other.shape);
  this->stride = std::move(other.stride);
  this->is_cuda = other.is_cuda;
  this->requires_grad = other.requires_grad;
  this->grad_tensor = other.grad_tensor;
  this->grad_fn = std::move(other.grad_fn);

  other.d_data = nullptr;
  other.grad_tensor = nullptr;
  other.size = 0;
  other.requires_grad = false;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
  if (this != &other)
  {
    if (this->d_data)
    {
      CudaMemoryPool::instance().deallocate(
          this->d_data,
          this->size * sizeof(float));
    }

    if (this->grad_tensor)
    {
      delete this->grad_tensor;
    }

    this->d_data = other.d_data;
    this->size = other.size;
    this->shape = std::move(other.shape);
    this->stride = std::move(other.stride);
    this->is_cuda = other.is_cuda;
    this->requires_grad = other.requires_grad;
    this->grad_tensor = other.grad_tensor;
    this->grad_fn = std::move(other.grad_fn);

    other.d_data = nullptr;
    other.grad_tensor = nullptr;
    other.size = 0;
    other.requires_grad = false;
  }

  return *this;
}

Tensor::~Tensor()
{
  if (d_data)
    CudaMemoryPool::instance().deallocate(d_data, size * sizeof(float));

  if (grad_tensor)
    delete grad_tensor;
}

std::vector<float> Tensor::cpu() const
{
  std::vector<float> host(size);

  cudaMemcpy(
      host.data(),
      d_data,
      size * sizeof(float),
      cudaMemcpyDeviceToHost);

  return host;
}

Tensor Tensor::operator+(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);
  Tensor out(out_shape, requires_grad || other.requires_grad);

  if (shape == other.shape)
  {
    launch_add(d_data, other.d_data, out.d_data, size);

    if (out.requires_grad)
    {
      out.grad_fn = std::make_shared<AutogradNode>(
          OpType::ADD,
          std::vector<Tensor *>{const_cast<Tensor *>(this),
                                const_cast<Tensor *>(&other)});
    }

    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_add_broadcast(
      d_data, other.d_data, out.d_data,
      d_a_shape, d_a_stride,
      d_b_shape, d_b_stride,
      d_out_shape,
      static_cast<int>(shape.size()),
      static_cast<int>(other.shape.size()),
      static_cast<int>(out_shape.size()),
      out.size);

  free_device_int_vector(d_a_shape, shape);
  free_device_int_vector(d_a_stride, stride);
  free_device_int_vector(d_b_shape, other.shape);
  free_device_int_vector(d_b_stride, other.stride);
  free_device_int_vector(d_out_shape, out_shape);

  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>(
        OpType::ADD,
        std::vector<Tensor *>{const_cast<Tensor *>(this),
                              const_cast<Tensor *>(&other)});
  }

  return out;
}

Tensor Tensor::operator-(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);
  Tensor out(out_shape, requires_grad || other.requires_grad);

  if (shape == other.shape)
  {
    launch_sub(d_data, other.d_data, out.d_data, size);

    if (out.requires_grad)
    {
      out.grad_fn = std::make_shared<AutogradNode>(
          OpType::SUB,
          std::vector<Tensor *>{const_cast<Tensor *>(this),
                                const_cast<Tensor *>(&other)});
    }

    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_sub_broadcast(
      d_data, other.d_data, out.d_data,
      d_a_shape, d_a_stride,
      d_b_shape, d_b_stride,
      d_out_shape,
      static_cast<int>(shape.size()),
      static_cast<int>(other.shape.size()),
      static_cast<int>(out_shape.size()),
      out.size);

  free_device_int_vector(d_a_shape, shape);
  free_device_int_vector(d_a_stride, stride);
  free_device_int_vector(d_b_shape, other.shape);
  free_device_int_vector(d_b_stride, other.stride);
  free_device_int_vector(d_out_shape, out_shape);

  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>(
        OpType::SUB,
        std::vector<Tensor *>{const_cast<Tensor *>(this),
                              const_cast<Tensor *>(&other)});
  }

  return out;
}

Tensor Tensor::operator*(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);
  Tensor out(out_shape, requires_grad || other.requires_grad);

  if (shape == other.shape)
  {
    launch_mul(d_data, other.d_data, out.d_data, size);

    if (out.requires_grad)
    {
      out.grad_fn = std::make_shared<AutogradNode>(
          OpType::MUL,
          std::vector<Tensor *>{const_cast<Tensor *>(this),
                                const_cast<Tensor *>(&other)});
    }

    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_mul_broadcast(
      d_data,
      other.d_data,
      out.d_data,
      d_a_shape,
      d_a_stride,
      d_b_shape,
      d_b_stride,
      d_out_shape,
      static_cast<int>(shape.size()),
      static_cast<int>(other.shape.size()),
      static_cast<int>(out_shape.size()),
      out.size);

  free_device_int_vector(d_a_shape, shape);
  free_device_int_vector(d_a_stride, stride);
  free_device_int_vector(d_b_shape, other.shape);
  free_device_int_vector(d_b_stride, other.stride);
  free_device_int_vector(d_out_shape, out_shape);

  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>(
        OpType::MUL,
        std::vector<Tensor *>{const_cast<Tensor *>(this),
                              const_cast<Tensor *>(&other)});
  }

  return out;
}

Tensor Tensor::operator/(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);
  Tensor out(out_shape, requires_grad || other.requires_grad);

  if (shape == other.shape)
  {
    launch_div(d_data, other.d_data, out.d_data, size);

    if (out.requires_grad)
    {
      out.grad_fn = std::make_shared<AutogradNode>(
          OpType::DIV,
          std::vector<Tensor *>{const_cast<Tensor *>(this),
                                const_cast<Tensor *>(&other)});
    }

    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_div_broadcast(
      d_data, other.d_data, out.d_data,
      d_a_shape, d_a_stride,
      d_b_shape, d_b_stride,
      d_out_shape,
      static_cast<int>(shape.size()),
      static_cast<int>(other.shape.size()),
      static_cast<int>(out_shape.size()),
      out.size);

  free_device_int_vector(d_a_shape, shape);
  free_device_int_vector(d_a_stride, stride);
  free_device_int_vector(d_b_shape, other.shape);
  free_device_int_vector(d_b_stride, other.stride);
  free_device_int_vector(d_out_shape, out_shape);

  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>(
        OpType::DIV,
        std::vector<Tensor *>{const_cast<Tensor *>(this),
                              const_cast<Tensor *>(&other)});
  }

  return out;
}

Tensor Tensor::matmul(const Tensor &other) const
{
  std::vector<int> out_shape = matmul_output_shape(shape, other.shape);
  Tensor out(out_shape, requires_grad || other.requires_grad);

  std::vector<int> a_batch = batch_shape_of_matmul(shape);
  std::vector<int> b_batch = batch_shape_of_matmul(other.shape);
  std::vector<int> out_batch = batch_shape_of_matmul(out_shape);

  int M = shape[shape.size() - 2];
  int K = shape[shape.size() - 1];
  int N = other.shape[other.shape.size() - 1];

  int out_batch_size = numel_from_shape(out_batch);

  if (shape.size() == 2 && other.shape.size() == 2)
  {
    launch_matmul(d_data, other.d_data, out.d_data, M, N, K);
  }
  else if (a_batch == b_batch)
  {
    launch_batched_matmul(
        d_data,
        other.d_data,
        out.d_data,
        M,
        N,
        K,
        out_batch_size);
  }
  else
  {
    std::vector<int> a_batch_stride = compute_stride_from_shape(a_batch);
    std::vector<int> b_batch_stride = compute_stride_from_shape(b_batch);

    int *d_a_batch_shape = copy_int_vector_to_device(a_batch);
    int *d_a_batch_stride = copy_int_vector_to_device(a_batch_stride);
    int *d_b_batch_shape = copy_int_vector_to_device(b_batch);
    int *d_b_batch_stride = copy_int_vector_to_device(b_batch_stride);
    int *d_out_batch_shape = copy_int_vector_to_device(out_batch);

    launch_broadcasted_batched_matmul(
        d_data,
        other.d_data,
        out.d_data,
        d_a_batch_shape,
        d_a_batch_stride,
        d_b_batch_shape,
        d_b_batch_stride,
        d_out_batch_shape,
        static_cast<int>(a_batch.size()),
        static_cast<int>(b_batch.size()),
        static_cast<int>(out_batch.size()),
        M,
        N,
        K,
        out_batch_size);

    free_device_int_vector(d_a_batch_shape, a_batch);
    free_device_int_vector(d_a_batch_stride, a_batch_stride);
    free_device_int_vector(d_b_batch_shape, b_batch);
    free_device_int_vector(d_b_batch_stride, b_batch_stride);
    free_device_int_vector(d_out_batch_shape, out_batch);
  }
  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>(
        OpType::MATMUL,
        std::vector<Tensor *>{const_cast<Tensor *>(this),
                              const_cast<Tensor *>(&other)});
  }

  return out;
}

std::vector<int> Tensor::get_shape() const
{
  return shape;
}

int Tensor::ndim() const
{
  return static_cast<int>(shape.size());
}

int Tensor::numel() const
{
  return size;
}

std::vector<int> Tensor::get_stride() const
{
  return stride;
}

Tensor Tensor::reshape(std::vector<int> new_shape) const
{
  int new_size = 1;

  for (int dim : new_shape)
    new_size *= dim;

  if (new_size != size)
    throw std::runtime_error("reshape size mismatch");

  Tensor out(new_shape, requires_grad);

  cudaMemcpy(
      out.d_data,
      d_data,
      size * sizeof(float),
      cudaMemcpyDeviceToDevice);

  return out;
}

Tensor Tensor::flatten() const
{
  return reshape(std::vector<int>{size});
}

Tensor Tensor::transpose() const
{
  if (shape.size() != 2)
    throw std::runtime_error("transpose currently supports only 2D tensors");

  int rows = shape[0];
  int cols = shape[1];

  Tensor out(std::vector<int>{cols, rows}, requires_grad);

  launch_transpose(d_data, out.d_data, rows, cols);

  return out;
}

Tensor Tensor::sum() const
{
  Tensor out(std::vector<int>{1}, requires_grad);

  launch_sum(d_data, out.d_data, size);

  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>();
    out.grad_fn->op = OpType::SUM;

    // If this tensor is an intermediate, save an owned copy
    // so expressions like (x*x).sum() do not point to a destroyed temporary.
    if (this->grad_fn)
    {
      Tensor saved(this->cpu(), this->shape, this->requires_grad);
      saved.grad_fn = this->grad_fn;

      out.grad_fn->saved_tensors.push_back(
          std::make_shared<Tensor>(std::move(saved)));

      out.grad_fn->parents.push_back(
          out.grad_fn->saved_tensors[0].get());
    }
    else
    {
      // Leaf tensor: safe to point directly to the user-owned tensor.
      out.grad_fn->parents.push_back(const_cast<Tensor *>(this));
    }
  }

  return out;
}

Tensor Tensor::mean() const
{
  Tensor out = sum();

  launch_divide_scalar(
      out.d_data,
      static_cast<float>(size));

  if (out.requires_grad)
  {
    out.grad_fn = std::make_shared<AutogradNode>();

    out.grad_fn->op = OpType::MEAN;

    if (this->grad_fn)
    {
      Tensor saved(this->cpu(), shape, requires_grad);
      saved.grad_fn = this->grad_fn;

      out.grad_fn->saved_tensors.push_back(
          std::make_shared<Tensor>(std::move(saved)));

      out.grad_fn->parents.push_back(
          out.grad_fn->saved_tensors[0].get());
    }
    else
    {
      out.grad_fn->parents.push_back(
          const_cast<Tensor *>(this));
    }
  }

  return out;
}

Tensor Tensor::max() const
{
  Tensor out(std::vector<int>{1}, requires_grad);

  launch_max(
      d_data,
      out.d_data,
      size);

  return out;
}

Tensor Tensor::transpose_last_two_dims() const
{
  if (shape.size() < 2)
    throw std::runtime_error("transpose_last_two_dims requires tensor with at least 2 dimensions");

  std::vector<int> out_shape = shape;

  int ndim = static_cast<int>(shape.size());

  std::swap(out_shape[ndim - 1], out_shape[ndim - 2]);

  Tensor out(out_shape, requires_grad);

  int *d_in_shape = copy_int_vector_to_device(shape);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_transpose_last_two_dims(
      d_data,
      out.d_data,
      d_in_shape,
      d_out_shape,
      ndim,
      out.size);

  free_device_int_vector(d_in_shape, shape);
  free_device_int_vector(d_out_shape, out_shape);

  return out;
}

Tensor Tensor::sum_to_shape(std::vector<int> target_shape) const
{
  if (target_shape.size() > shape.size())
    throw std::runtime_error("sum_to_shape target shape cannot have more dimensions than input shape");

  Tensor out(target_shape, requires_grad);

  // Important: output must start at zero because kernel uses atomicAdd.
  launch_fill(out.d_data, 0.0f, out.size);

  std::vector<int> out_stride(target_shape.size());

  int running_stride = 1;
  for (int i = static_cast<int>(target_shape.size()) - 1; i >= 0; --i)
  {
    out_stride[i] = running_stride;
    running_stride *= target_shape[i];
  }

  int *d_in_shape = copy_int_vector_to_device(shape);
  int *d_out_shape = copy_int_vector_to_device(target_shape);
  int *d_out_stride = copy_int_vector_to_device(out_stride);

  launch_sum_to_shape(
      d_data,
      out.d_data,
      d_in_shape,
      d_out_shape,
      d_out_stride,
      static_cast<int>(shape.size()),
      static_cast<int>(target_shape.size()),
      size);

  free_device_int_vector(d_in_shape, shape);
  free_device_int_vector(d_out_shape, target_shape);
  free_device_int_vector(d_out_stride, out_stride);

  return out;
}

Tensor Tensor::clone() const
{
  std::vector<float> host = cpu();

  Tensor out(host, shape, requires_grad);

  return out;
}
