#include <cuda_runtime.h>
#include <cstdio>

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
