#include "tensor.h"
#include "memory_pool.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

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
  if (shape != other.shape)
    throw std::runtime_error("Addition requires tensors to be of same shape");
  Tensor out(shape);
  launch_add(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::operator*(const Tensor &other) const
{
  if (shape != other.shape)
    throw std::runtime_error("Multiplication requires tensors to be of same shape");
  Tensor out(shape);
  launch_mul(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::operator-(const Tensor &other) const
{
  if (shape != other.shape)
    throw std::runtime_error("Subtraction requires tensors to be of same shape");
  Tensor out(shape);
  launch_sub(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::operator/(const Tensor &other) const
{
  if (shape != other.shape)
    throw std::runtime_error("Division requires tensors to be of same shape");
  Tensor out(shape);
  launch_div(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::matmul(const Tensor &other) const
{
  if (shape.size() != 2 || other.shape.size() != 2)
    throw std::runtime_error("matmul currently supports only 2D tensors");

  int M = shape[0];
  int K = shape[1];

  int K_other = other.shape[0];
  int N = other.shape[1];

  if (K != K_other)
    throw std::runtime_error("matmul shape mismatch");

  Tensor out(std::vector<int>{M, N});

  launch_matmul(d_data, other.d_data, out.d_data, M, N, K);

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
