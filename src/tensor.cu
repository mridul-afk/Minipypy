#include <cuda_runtime.h>

__global__ void add_kernel(const float *a, const float *b, float *out, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = a[idx] + b[idx];
}

__global__ void mul_kernel(const float *a, const float *b, float *out, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = a[idx] * b[idx];
}

__global__ void sub_kernel(const float *a, const float *b, float *out, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = a[idx] - b[idx];
}

__global__ void div_kernel(const float *a, const float *b, float *out, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = a[idx] / b[idx];
}

void launch_add(const float *a, const float *b, float *out, int size)
{
  add_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
  cudaDeviceSynchronize();
}

void launch_mul(const float *a, const float *b, float *out, int size)
{
  mul_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
  cudaDeviceSynchronize();
}

void launch_sub(const float *a, const float *b, float *out, int size)
{
  sub_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
  cudaDeviceSynchronize();
}

void launch_div(const float *a, const float *b, float *out, int size)
{
  div_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
  cudaDeviceSynchronize();
}
