#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <vector>

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

__global__ void matmul_kernel(
    const float *a,
    const float *b,
    float *out,
    int M,
    int N,
    int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    float sum = 0.0f;

    for (int i = 0; i < K; i++)
    {
      sum += a[row * K + i] * b[i * N + col];
    }

    out[row * N + col] = sum;
  }
}

__global__ void transpose_kernel(
    const float *input,
    float *output,
    int rows,
    int cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && col < cols)
  {
    output[col * rows + row] = input[row * cols + col];
  }
}

__global__ void sum_kernel(const float *input, float *output, int size)
{
  __shared__ float shared[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float value = 0.0f;

  if (idx < size)
    value = input[idx];

  shared[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
      shared[tid] += shared[tid + stride];

    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(output, shared[0]);
}

__global__ void divide_scalar_kernel(
    float *data,
    float scalar)
{
  data[0] /= scalar;
}

__global__ void max_kernel(
    const float *input,
    float *output,
    int size)
{
  __shared__ float shared[256];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float value = -FLT_MAX;

  if (idx < size)
    value = input[idx];

  shared[tid] = value;

  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
    {
      if (shared[tid + stride] > shared[tid])
        shared[tid] = shared[tid + stride];
    }

    __syncthreads();
  }

  if (tid == 0)
    atomicMax((int *)output, __float_as_int(shared[0]));
}

void launch_add(const float *a, const float *b, float *out, int size)
{
  add_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void launch_mul(const float *a, const float *b, float *out, int size)
{
  mul_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void launch_sub(const float *a, const float *b, float *out, int size)
{
  sub_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void launch_div(const float *a, const float *b, float *out, int size)
{
  div_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void launch_matmul(const float *a, const float *b, float *out, int M, int N, int K)
{
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y);

  matmul_kernel<<<grid, block>>>(a, b, out, M, N, K);
}

void launch_transpose(const float *input, float *output, int rows, int cols)
{
  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x,
            (rows + block.y - 1) / block.y);

  transpose_kernel<<<grid, block>>>(input, output, rows, cols);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("transpose kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

void launch_sum(const float *input, float *output, int size)
{
  cudaMemset(output, 0, sizeof(float));

  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  sum_kernel<<<blocks, threads>>>(input, output, size);
}

void launch_divide_scalar(
    float *data,
    float scalar)
{
  divide_scalar_kernel<<<1, 1>>>(data, scalar);
}

void launch_max(const float *input, float *output, int size)
{
  std::vector<float> host(size);

  cudaMemcpy(
      host.data(),
      input,
      size * sizeof(float),
      cudaMemcpyDeviceToHost);

  float max_val = host[0];

  for (int i = 1; i < size; i++)
  {
    if (host[i] > max_val)
      max_val = host[i];
  }

  cudaMemcpy(
      output,
      &max_val,
      sizeof(float),
      cudaMemcpyHostToDevice);
}
