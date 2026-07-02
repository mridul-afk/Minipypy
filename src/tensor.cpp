#include "tensor.h"
#include "memory_pool.h"

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

void launch_batched_matmul(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_size);

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

Tensor::Tensor(int size)
{
  this->size = size;
  this->shape = {size};
  this->stride = {1};
  this->is_cuda = true;
  this->d_data = nullptr;

  d_data = static_cast<float *>(
      CudaMemoryPool::instance().allocate(size * sizeof(float)));
}

Tensor::Tensor(const std::vector<float> &host_data)
{
  this->size = static_cast<int>(host_data.size());
  this->shape = {size};
  this->stride = {1};
  this->is_cuda = true;
  this->d_data = nullptr;

  d_data = static_cast<float *>(
      CudaMemoryPool::instance().allocate(size * sizeof(float)));

  cudaMemcpy(
      d_data,
      host_data.data(),
      size * sizeof(float),
      cudaMemcpyHostToDevice);
}

Tensor::Tensor(std::vector<int> shape)
{
  this->shape = shape;
  this->size = 1;
  this->is_cuda = true;
  this->d_data = nullptr;

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

Tensor::Tensor(const std::vector<float> &host_data, std::vector<int> shape)
{
  this->shape = shape;
  this->size = 1;
  this->is_cuda = true;
  this->d_data = nullptr;

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

// Disable copy in tensor.h, enable move.
Tensor::Tensor(Tensor &&other) noexcept
{
  this->d_data = other.d_data;
  this->size = other.size;
  this->shape = std::move(other.shape);
  this->stride = std::move(other.stride);
  this->is_cuda = other.is_cuda;

  other.d_data = nullptr;
  other.size = 0;
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

    this->d_data = other.d_data;
    this->size = other.size;
    this->shape = std::move(other.shape);
    this->stride = std::move(other.stride);
    this->is_cuda = other.is_cuda;

    other.d_data = nullptr;
    other.size = 0;
  }

  return *this;
}

Tensor::~Tensor()
{
  if (d_data)
    CudaMemoryPool::instance().deallocate(d_data, size * sizeof(float));
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

  Tensor out(out_shape);

  // Fast path: same shape
  if (shape == other.shape)
  {
    launch_add(d_data, other.d_data, out.d_data, size);
    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_add_broadcast(
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

  return out;
}

Tensor Tensor::operator-(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);

  Tensor out(out_shape);

  // Fast path: same shape
  if (shape == other.shape)
  {
    launch_sub(d_data, other.d_data, out.d_data, size);
    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_sub_broadcast(
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

  return out;
}

Tensor Tensor::operator*(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);

  Tensor out(out_shape);

  // Fast path: same shape
  if (shape == other.shape)
  {
    launch_mul(d_data, other.d_data, out.d_data, size);
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

  return out;
}

Tensor Tensor::operator/(const Tensor &other) const
{
  std::vector<int> out_shape = broadcast_shape(shape, other.shape);

  Tensor out(out_shape);

  // Fast path: same shape
  if (shape == other.shape)
  {
    launch_div(d_data, other.d_data, out.d_data, size);
    return out;
  }

  int *d_a_shape = copy_int_vector_to_device(shape);
  int *d_a_stride = copy_int_vector_to_device(stride);
  int *d_b_shape = copy_int_vector_to_device(other.shape);
  int *d_b_stride = copy_int_vector_to_device(other.stride);
  int *d_out_shape = copy_int_vector_to_device(out_shape);

  launch_div_broadcast(
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

  return out;
}

Tensor Tensor::matmul(const Tensor &other) const
{
  if (shape.size() < 2 || other.shape.size() < 2)
    throw std::runtime_error("NEW ND MATMUL FUNCTION IS ACTIVE");

  int ndim = static_cast<int>(shape.size());
  int other_ndim = static_cast<int>(other.shape.size());

  if (ndim != other_ndim)
    throw std::runtime_error("N-D matmul currently requires same number of dimensions");

  // Check batch dimensions
  for (int i = 0; i < ndim - 2; i++)
  {
    if (shape[i] != other.shape[i])
      throw std::runtime_error("N-D matmul batch dimensions must match");
  }

  int M = shape[ndim - 2];
  int K = shape[ndim - 1];

  int K_other = other.shape[other_ndim - 2];
  int N = other.shape[other_ndim - 1];

  if (K != K_other)
    throw std::runtime_error("matmul shape mismatch");

  std::vector<int> out_shape;

  for (int i = 0; i < ndim - 2; i++)
    out_shape.push_back(shape[i]);

  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor out(out_shape);

  if (ndim == 2)
  {
    launch_matmul(d_data, other.d_data, out.d_data, M, N, K);
  }
  else
  {
    int batch_size = 1;

    for (int i = 0; i < ndim - 2; i++)
      batch_size *= shape[i];

    launch_batched_matmul(
        d_data,
        other.d_data,
        out.d_data,
        M,
        N,
        K,
        batch_size);
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

  Tensor out(new_shape);

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

  Tensor out(std::vector<int>{cols, rows});

  launch_transpose(d_data, out.d_data, rows, cols);

  return out;
}

Tensor Tensor::sum() const
{
  Tensor out(std::vector<int>{1});

  launch_sum(d_data, out.d_data, size);

  return out;
}

Tensor Tensor::mean() const
{
  Tensor out = sum();

  launch_divide_scalar(out.d_data,
                       static_cast<float>(size));

  return out;
}

Tensor Tensor::max() const
{
  Tensor out(std::vector<int>{1});

  launch_max(
      d_data,
      out.d_data,
      size);

  return out;
}
