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
