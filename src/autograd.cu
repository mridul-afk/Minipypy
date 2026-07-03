#include <cuda_runtime.h>

__global__ void fill_kernel(float *data, float value, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    data[idx] = value;
}

void launch_fill(float *data, float value, int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  fill_kernel<<<blocks, threads>>>(data, value, size);
}

__global__ void mul_backward_kernel(
    const float *grad_out,
    const float *other,
    float *grad_in,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    grad_in[idx] += grad_out[idx] * other[idx];
}

void launch_mul_backward(
    const float *grad_out,
    const float *other,
    float *grad_in,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  mul_backward_kernel<<<blocks, threads>>>(
      grad_out,
      other,
      grad_in,
      size);
}
__global__ void sum_backward_kernel(
    const float *grad_out,
    float *grad_in,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    grad_in[idx] += grad_out[0];
}

void launch_sum_backward(
    const float *grad_out,
    float *grad_in,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  sum_backward_kernel<<<blocks, threads>>>(
      grad_out,
      grad_in,
      size);
}
