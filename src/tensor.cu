#include <cuda_runtime.h>
#include <cstdio>
#include <cfloat>
#include <vector>
#include <float.h>
#include <math.h>

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

__global__ void transpose_last_two_dims_kernel(
    const float *input,
    float *output,
    const int *in_shape,
    const int *out_shape,
    int ndim,
    int total_size)
{
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (out_idx >= total_size)
    return;

  int remaining = out_idx;

  int in_idx = 0;

  // We decode output index -> output coordinates.
  // Then map output coordinates to input coordinates,
  // swapping only the last two dimensions.
  for (int d = ndim - 1; d >= 0; --d)
  {
    int coord = remaining % out_shape[d];
    remaining /= out_shape[d];

    int in_dim;

    if (d == ndim - 1)
    {
      // output last dim corresponds to input second-last dim
      in_dim = ndim - 2;
    }
    else if (d == ndim - 2)
    {
      // output second-last dim corresponds to input last dim
      in_dim = ndim - 1;
    }
    else
    {
      // batch dims stay the same
      in_dim = d;
    }

    int input_stride = 1;
    for (int s = in_dim + 1; s < ndim; ++s)
      input_stride *= in_shape[s];

    in_idx += coord * input_stride;
  }

  output[out_idx] = input[in_idx];
}

__global__ void sum_to_shape_kernel(
    const float *input,
    float *output,
    const int *in_shape,
    const int *out_shape,
    const int *out_stride,
    int in_ndim,
    int out_ndim,
    int input_size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= input_size)
    return;

  int remaining = idx;
  int out_idx = 0;

  int offset = in_ndim - out_ndim;

  // Decode input index into coordinates from right to left.
  for (int d = in_ndim - 1; d >= 0; --d)
  {
    int coord = remaining % in_shape[d];
    remaining /= in_shape[d];

    int out_d = d - offset;

    // Extra leading input dimensions are reduced away.
    if (out_d < 0)
      continue;

    // If output dim is 1, this input dim was broadcast.
    // So all input coords map to output coord 0.
    int out_coord = 0;

    if (out_shape[out_d] != 1)
      out_coord = coord;

    out_idx += out_coord * out_stride[out_d];
  }

  atomicAdd(&output[out_idx], input[idx]);
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

void launch_transpose_last_two_dims(
    const float *input,
    float *output,
    const int *in_shape,
    const int *out_shape,
    int ndim,
    int total_size)
{
  int threads = 256;
  int blocks = (total_size + threads - 1) / threads;

  transpose_last_two_dims_kernel<<<blocks, threads>>>(
      input,
      output,
      in_shape,
      out_shape,
      ndim,
      total_size);
}

void launch_sum_to_shape(
    const float *input,
    float *output,
    const int *in_shape,
    const int *out_shape,
    const int *out_stride,
    int in_ndim,
    int out_ndim,
    int input_size)
{
  int threads = 256;
  int blocks = (input_size + threads - 1) / threads;

  sum_to_shape_kernel<<<blocks, threads>>>(
      input,
      output,
      in_shape,
      out_shape,
      out_stride,
      in_ndim,
      out_ndim,
      input_size);
}

// TENSOR-SCALAR OPS

__global__ void mul_scalar_kernel(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    out[idx] = a[idx] * scalar;
}

void launch_mul_scalar(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  mul_scalar_kernel<<<blocks, threads>>>(
      a,
      out,
      scalar,
      size);
}

__global__ void add_scalar_kernel(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    out[idx] = a[idx] + scalar;
}

void launch_add_scalar(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  add_scalar_kernel<<<blocks, threads>>>(
      a,
      out,
      scalar,
      size);
}

__global__ void sub_scalar_kernel(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    out[idx] = a[idx] - scalar;
}

void launch_sub_scalar(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  sub_scalar_kernel<<<blocks, threads>>>(
      a,
      out,
      scalar,
      size);
}

__global__ void rsub_scalar_kernel(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    out[idx] = scalar - a[idx];
}

void launch_rsub_scalar(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  rsub_scalar_kernel<<<blocks, threads>>>(
      a,
      out,
      scalar,
      size);
}

__global__ void div_scalar_kernel(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    out[idx] = a[idx] / scalar;
}

void launch_div_scalar(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  div_scalar_kernel<<<blocks, threads>>>(
      a,
      out,
      scalar,
      size);
}

__global__ void rdiv_scalar_kernel(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
    out[idx] = scalar / a[idx];
}

void launch_rdiv_scalar(
    const float *a,
    float *out,
    float scalar,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  rdiv_scalar_kernel<<<blocks, threads>>>(
      a,
      out,
      scalar,
      size);
}

__global__ void relu_forward_kernel(
    const float *a,
    float *out,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
  {
    out[idx] = a[idx] > 0.0f ? a[idx] : 0.0f;
  }
}

void launch_relu_forward(
    const float *a,
    float *out,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  relu_forward_kernel<<<blocks, threads>>>(a, out, size);
}

__global__ void softmax_forward_kernel(
    const float *input,
    float *output,
    int outer_size,
    int softmax_size,
    int inner_size)
{
  /*
    N-D contiguous softmax.

    For a tensor shape:
      [d0, d1, ..., dn]

    softmax(dim) is flattened into:

      outer_size   = product of dims before dim
      softmax_size = shape[dim]
      inner_size   = product of dims after dim

    Each CUDA block handles one softmax group.

    Example:
      shape = [2, 3, 4]
      dim = 1

      outer_size   = 2
      softmax_size = 3
      inner_size   = 4

    There are outer_size * inner_size groups.
    Each group has softmax_size elements.
  */

  int group = blockIdx.x;
  int total_groups = outer_size * inner_size;

  if (group >= total_groups)
    return;

  int outer_idx = group / inner_size;
  int inner_idx = group % inner_size;

  int base = outer_idx * softmax_size * inner_size + inner_idx;

  /*
    Step 1:
    Find max value inside this softmax group.

    This makes softmax numerically stable:
      exp(x_i - max(x))
  */
  float local_max = -FLT_MAX;

  for (int j = threadIdx.x; j < softmax_size; j += blockDim.x)
  {
    int idx = base + j * inner_size;
    float v = input[idx];

    if (v > local_max)
      local_max = v;
  }

  __shared__ float shared_max[256];
  shared_max[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (threadIdx.x < stride)
    {
      if (shared_max[threadIdx.x + stride] > shared_max[threadIdx.x])
        shared_max[threadIdx.x] = shared_max[threadIdx.x + stride];
    }

    __syncthreads();
  }

  float group_max = shared_max[0];

  /*
    Step 2:
    Compute exp(x - max) and sum.
  */
  float local_sum = 0.0f;

  for (int j = threadIdx.x; j < softmax_size; j += blockDim.x)
  {
    int idx = base + j * inner_size;
    float e = expf(input[idx] - group_max);

    output[idx] = e;
    local_sum += e;
  }

  __shared__ float shared_sum[256];
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (threadIdx.x < stride)
    {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }

    __syncthreads();
  }

  float group_sum = shared_sum[0];

  /*
    Step 3:
    Normalize.
  */
  for (int j = threadIdx.x; j < softmax_size; j += blockDim.x)
  {
    int idx = base + j * inner_size;
    output[idx] = output[idx] / group_sum;
  }
}

void launch_softmax_forward(
    const float *input,
    float *output,
    int outer_size,
    int softmax_size,
    int inner_size)
{
  int threads = 256;
  int blocks = outer_size * inner_size;

  softmax_forward_kernel<<<blocks, threads>>>(
      input,
      output,
      outer_size,
      softmax_size,
      inner_size);
}

__global__ void cross_entropy_forward_kernel(
    const float *logits,
    const float *target,
    float *loss,
    int batch_size,
    int num_classes)
{
  /*
  Stable CrossEntropyLoss forward.

  logits shape:
    [batch_size, num_classes]

  target shape:
    [batch_size] or [batch_size, 1]

  For each sample i:
    loss_i = -logit[target_i] + max_logit
             + log(sum_j exp(logit_j - max_logit))

  Final output:
    mean(loss_i)
*/

  int sample = blockIdx.x;

  if (sample >= batch_size)
    return;

  int tid = threadIdx.x;
  int base = sample * num_classes;

  int label = static_cast<int>(target[sample]);
  /*
    Step 1:
    Find max logit for numerical stability.
  */
  float local_max = -FLT_MAX;

  for (int j = tid; j < num_classes; j += blockDim.x)
  {
    float v = logits[base + j];

    if (v > local_max)
      local_max = v;
  }

  __shared__ float shared_max[256];
  shared_max[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      if (shared_max[tid + stride] > shared_max[tid])
        shared_max[tid] = shared_max[tid + stride];
    }

    __syncthreads();
  }

  float max_logit = shared_max[0];

  /*
    Step 2:
    Compute sum_j exp(logit_j - max_logit)
  */
  float local_sum = 0.0f;

  for (int j = tid; j < num_classes; j += blockDim.x)
  {
    local_sum += expf(logits[base + j] - max_logit);
  }

  __shared__ float shared_sum[256];
  shared_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      shared_sum[tid] += shared_sum[tid + stride];
    }

    __syncthreads();
  }

  float sum_exp = shared_sum[0];

  /*
    Step 3:
    Compute sample loss and add mean contribution.
  */
  if (tid == 0)
  {
    float correct_logit = logits[base + label];

    float sample_loss =
        -correct_logit + max_logit + logf(sum_exp);

    atomicAdd(loss, sample_loss / static_cast<float>(batch_size));
  }
}

void launch_cross_entropy_forward(
    const float *logits,
    const float *target,
    float *loss,
    int batch_size,
    int num_classes)
{
  cudaMemset(loss, 0, sizeof(float));

  int threads = 256;
  int blocks = batch_size;

  cross_entropy_forward_kernel<<<blocks, threads>>>(
      logits,
      target,
      loss,
      batch_size,
      num_classes);
}

__global__ void bce_with_logits_forward_kernel(
    const float *logits,
    const float *target,
    float *loss,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= size)
    return;

  float x = logits[idx];
  float y = target[idx];

  float max_val = x > 0.0f ? x : 0.0f;
  float abs_x = fabsf(x);

  float element_loss = max_val - x * y + logf(1.0f + expf(-abs_x));

  atomicAdd(loss, element_loss / static_cast<float>(size));
}

void launch_bce_with_logits_forward(
    const float *logits,
    const float *target,
    float *loss,
    int size)
{
  cudaMemset(loss, 0, sizeof(float));

  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  bce_with_logits_forward_kernel<<<blocks, threads>>>(
      logits,
      target,
      loss,
      size);
}

__global__ void sqrt_forward_kernel(
    const float *input,
    float *output,
    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size)
  {
    output[idx] = sqrtf(input[idx]);
  }
}

void launch_sqrt_forward(
    const float *input,
    float *output,
    int size)
{
  int threads = 256;
  int blocks = (size + threads - 1) / threads;

  sqrt_forward_kernel<<<blocks, threads>>>(
      input,
      output,
      size);
}
