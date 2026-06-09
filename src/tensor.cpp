#include "tensor.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>

// CUDA launchers from tensor.cu
void launch_add(const float *a, const float *b, float *out, int size);
void launch_mul(const float *a, const float *b, float *out, int size);
void launch_sub(const float *a, const float *b, float *out, int size);
void launch_div(const float *a, const float *b, float *out, int size);

Tensor::Tensor(int size)
{
  this->size = size;
  this->shape = {size};
  this->stride = {1};
  this->is_cuda = true;
  this->d_data = nullptr;

  cudaMalloc(&d_data, size * sizeof(float));
}

Tensor::Tensor(const std::vector<float> &host_data)
{
  this->size = static_cast<int>(host_data.size());
  this->shape = {size};
  this->stride = {1};
  this->is_cuda = true;
  this->d_data = nullptr;

  cudaMalloc(&d_data, size * sizeof(float));

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

  cudaMalloc(&d_data, size * sizeof(float));
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
      cudaFree(this->d_data);

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
    cudaFree(d_data);
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
  Tensor out(size);
  launch_add(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::operator*(const Tensor &other) const
{
  Tensor out(size);
  launch_mul(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::operator-(const Tensor &other) const
{
  Tensor out(size);
  launch_sub(d_data, other.d_data, out.d_data, size);
  return out;
}

Tensor Tensor::operator/(const Tensor &other) const
{
  Tensor out(size);
  launch_div(d_data, other.d_data, out.d_data, size);
  return out;
}
