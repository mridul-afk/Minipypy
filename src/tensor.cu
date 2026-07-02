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

constexpr int TILE_SIZE = 16;

__global__ void matmul_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K)
{
  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
  __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < num_tiles; tile++)
  {
    int a_col = tile * TILE_SIZE + threadIdx.x;
    int b_row = tile * TILE_SIZE + threadIdx.y;

    sh_A[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

    sh_B[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++)
      sum += sh_A[threadIdx.y][i] * sh_B[i][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
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

__global__ void add_broadcast_kernel(
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
    int out_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= out_size)
    return;

  int temp = idx;
  int a_offset = 0;
  int b_offset = 0;

  for (int dim = out_ndim - 1; dim >= 0; dim--)
  {
    int coord = temp % out_shape[dim];
    temp /= out_shape[dim];

    int a_dim = dim - (out_ndim - a_ndim);
    int b_dim = dim - (out_ndim - b_ndim);

    if (a_dim >= 0)
    {
      int a_coord = (a_shape[a_dim] == 1) ? 0 : coord;
      a_offset += a_coord * a_stride[a_dim];
    }

    if (b_dim >= 0)
    {
      int b_coord = (b_shape[b_dim] == 1) ? 0 : coord;
      b_offset += b_coord * b_stride[b_dim];
    }
  }

  out[idx] = a[a_offset] + b[b_offset];
}

__global__ void sub_broadcast_kernel(
    const float *a, const float *b, float *out,
    const int *a_shape, const int *a_stride,
    const int *b_shape, const int *b_stride,
    const int *out_shape,
    int a_ndim, int b_ndim, int out_ndim, int out_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_size)
    return;

  int temp = idx;
  int a_offset = 0;
  int b_offset = 0;

  for (int dim = out_ndim - 1; dim >= 0; dim--)
  {
    int coord = temp % out_shape[dim];
    temp /= out_shape[dim];

    int a_dim = dim - (out_ndim - a_ndim);
    int b_dim = dim - (out_ndim - b_ndim);

    if (a_dim >= 0)
      a_offset += ((a_shape[a_dim] == 1) ? 0 : coord) * a_stride[a_dim];

    if (b_dim >= 0)
      b_offset += ((b_shape[b_dim] == 1) ? 0 : coord) * b_stride[b_dim];
  }

  out[idx] = a[a_offset] - b[b_offset];
}

__global__ void mul_broadcast_kernel(
    const float *a, const float *b, float *out,
    const int *a_shape, const int *a_stride,
    const int *b_shape, const int *b_stride,
    const int *out_shape,
    int a_ndim, int b_ndim, int out_ndim, int out_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_size)
    return;

  int temp = idx;
  int a_offset = 0;
  int b_offset = 0;

  for (int dim = out_ndim - 1; dim >= 0; dim--)
  {
    int coord = temp % out_shape[dim];
    temp /= out_shape[dim];

    int a_dim = dim - (out_ndim - a_ndim);
    int b_dim = dim - (out_ndim - b_ndim);

    if (a_dim >= 0)
      a_offset += ((a_shape[a_dim] == 1) ? 0 : coord) * a_stride[a_dim];

    if (b_dim >= 0)
      b_offset += ((b_shape[b_dim] == 1) ? 0 : coord) * b_stride[b_dim];
  }

  out[idx] = a[a_offset] * b[b_offset];
}

__global__ void div_broadcast_kernel(
    const float *a, const float *b, float *out,
    const int *a_shape, const int *a_stride,
    const int *b_shape, const int *b_stride,
    const int *out_shape,
    int a_ndim, int b_ndim, int out_ndim, int out_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_size)
    return;

  int temp = idx;
  int a_offset = 0;
  int b_offset = 0;

  for (int dim = out_ndim - 1; dim >= 0; dim--)
  {
    int coord = temp % out_shape[dim];
    temp /= out_shape[dim];

    int a_dim = dim - (out_ndim - a_ndim);
    int b_dim = dim - (out_ndim - b_ndim);

    if (a_dim >= 0)
      a_offset += ((a_shape[a_dim] == 1) ? 0 : coord) * a_stride[a_dim];

    if (b_dim >= 0)
      b_offset += ((b_shape[b_dim] == 1) ? 0 : coord) * b_stride[b_dim];
  }

  out[idx] = a[a_offset] / b[b_offset];
}

__global__ void batched_matmul_kernel(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_size)
{
  int batch = blockIdx.z;

  if (batch >= batch_size)
    return;

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  const float *A_batch = A + batch * M * K;
  const float *B_batch = B + batch * K * N;
  float *C_batch = C + batch * M * N;

  __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
  __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < num_tiles; tile++)
  {
    int a_col = tile * TILE_SIZE + threadIdx.x;
    int b_row = tile * TILE_SIZE + threadIdx.y;

    sh_A[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A_batch[row * K + a_col] : 0.0f;

    sh_B[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B_batch[b_row * N + col] : 0.0f;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++)
      sum += sh_A[threadIdx.y][i] * sh_B[i][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    C_batch[row * N + col] = sum;
}

__device__ int compute_broadcast_batch_index(
    int out_batch,
    const int *input_batch_shape,
    const int *input_batch_stride,
    const int *out_batch_shape,
    int input_batch_ndim,
    int out_batch_ndim)
{
  int temp = out_batch;
  int input_batch_index = 0;

  for (int dim = out_batch_ndim - 1; dim >= 0; dim--)
  {
    int coord = temp % out_batch_shape[dim];
    temp /= out_batch_shape[dim];

    int input_dim = dim - (out_batch_ndim - input_batch_ndim);

    if (input_dim >= 0)
    {
      int input_coord =
          (input_batch_shape[input_dim] == 1) ? 0 : coord;

      input_batch_index += input_coord * input_batch_stride[input_dim];
    }
  }

  return input_batch_index;
}

__global__ void broadcasted_batched_matmul_kernel(
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
    int out_batch_size)
{
  int out_batch = blockIdx.z;

  if (out_batch >= out_batch_size)
    return;

  int a_batch_index = compute_broadcast_batch_index(
      out_batch,
      a_batch_shape,
      a_batch_stride,
      out_batch_shape,
      a_batch_ndim,
      out_batch_ndim);

  int b_batch_index = compute_broadcast_batch_index(
      out_batch,
      b_batch_shape,
      b_batch_stride,
      out_batch_shape,
      b_batch_ndim,
      out_batch_ndim);

  const float *A_batch = A + a_batch_index * M * K;
  const float *B_batch = B + b_batch_index * K * N;
  float *C_batch = C + out_batch * M * N;

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
  __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

  float sum = 0.0f;

  int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

  for (int tile = 0; tile < num_tiles; tile++)
  {
    int a_col = tile * TILE_SIZE + threadIdx.x;
    int b_row = tile * TILE_SIZE + threadIdx.y;

    sh_A[threadIdx.y][threadIdx.x] =
        (row < M && a_col < K) ? A_batch[row * K + a_col] : 0.0f;

    sh_B[threadIdx.y][threadIdx.x] =
        (b_row < K && col < N) ? B_batch[b_row * N + col] : 0.0f;

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++)
      sum += sh_A[threadIdx.y][i] * sh_B[i][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    C_batch[row * N + col] = sum;
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
    int out_size)
{
  int threads = 256;
  int blocks = (out_size + threads - 1) / threads;

  add_broadcast_kernel<<<blocks, threads>>>(
      a,
      b,
      out,
      a_shape,
      a_stride,
      b_shape,
      b_stride,
      out_shape,
      a_ndim,
      b_ndim,
      out_ndim,
      out_size);
}

void launch_sub_broadcast(
    const float *a, const float *b, float *out,
    const int *a_shape, const int *a_stride,
    const int *b_shape, const int *b_stride,
    const int *out_shape,
    int a_ndim, int b_ndim, int out_ndim, int out_size)
{
  int threads = 256;
  int blocks = (out_size + threads - 1) / threads;

  sub_broadcast_kernel<<<blocks, threads>>>(
      a, b, out,
      a_shape, a_stride,
      b_shape, b_stride,
      out_shape,
      a_ndim, b_ndim, out_ndim, out_size);
}

void launch_mul_broadcast(
    const float *a, const float *b, float *out,
    const int *a_shape, const int *a_stride,
    const int *b_shape, const int *b_stride,
    const int *out_shape,
    int a_ndim, int b_ndim, int out_ndim, int out_size)
{
  int threads = 256;
  int blocks = (out_size + threads - 1) / threads;

  mul_broadcast_kernel<<<blocks, threads>>>(
      a, b, out,
      a_shape, a_stride,
      b_shape, b_stride,
      out_shape,
      a_ndim, b_ndim, out_ndim, out_size);
}

void launch_div_broadcast(
    const float *a, const float *b, float *out,
    const int *a_shape, const int *a_stride,
    const int *b_shape, const int *b_stride,
    const int *out_shape,
    int a_ndim, int b_ndim, int out_ndim, int out_size)
{
  int threads = 256;
  int blocks = (out_size + threads - 1) / threads;

  div_broadcast_kernel<<<blocks, threads>>>(
      a, b, out,
      a_shape, a_stride,
      b_shape, b_stride,
      out_shape,
      a_ndim, b_ndim, out_ndim, out_size);
}

void launch_batched_matmul(
    const float *A,
    const float *B,
    float *C,
    int M,
    int N,
    int K,
    int batch_size)
{
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE,
            batch_size);

  batched_matmul_kernel<<<grid, block>>>(
      A,
      B,
      C,
      M,
      N,
      K,
      batch_size);
}

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
    int out_batch_size)
{
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE,
            out_batch_size);

  broadcasted_batched_matmul_kernel<<<grid, block>>>(
      A,
      B,
      C,
      a_batch_shape,
      a_batch_stride,
      b_batch_shape,
      b_batch_stride,
      out_batch_shape,
      a_batch_ndim,
      b_batch_ndim,
      out_batch_ndim,
      M,
      N,
      K,
      out_batch_size);
}
