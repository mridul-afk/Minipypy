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

__global__ void add_backward_kernel(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad_in[idx] += grad_out[idx];
}

void launch_add_backward(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    add_backward_kernel<<<blocks, threads>>>(
        grad_out,
        grad_in,
        size);
}

__global__ void sub_backward_left_kernel(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad_in[idx] += grad_out[idx];
}

void launch_sub_backward_left(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    sub_backward_left_kernel<<<blocks, threads>>>(
        grad_out,
        grad_in,
        size);
}

__global__ void sub_backward_right_kernel(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad_in[idx] -= grad_out[idx];
}

void launch_sub_backward_right(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    sub_backward_right_kernel<<<blocks, threads>>>(
        grad_out,
        grad_in,
        size);
}

__global__ void div_backward_left_kernel(
    const float *grad_out,
    const float *b,
    float *grad_a,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        grad_a[idx] += grad_out[idx] / b[idx];
}

void launch_div_backward_left(
    const float *grad_out,
    const float *b,
    float *grad_a,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    div_backward_left_kernel<<<blocks, threads>>>(
        grad_out,
        b,
        grad_a,
        size);
}

__global__ void div_backward_right_kernel(
    const float *grad_out,
    const float *a,
    const float *b,
    float *grad_b,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        grad_b[idx] += -grad_out[idx] * a[idx] / (b[idx] * b[idx]);
}

void launch_div_backward_right(
    const float *grad_out,
    const float *a,
    const float *b,
    float *grad_b,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    div_backward_right_kernel<<<blocks, threads>>>(
        grad_out,
        a,
        b,
        grad_b,
        size);
}

__global__ void mean_backward_kernel(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
        grad_in[idx] += grad_out[0] / size;
}

void launch_mean_backward(
    const float *grad_out,
    float *grad_in,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    mean_backward_kernel<<<blocks, threads>>>(
        grad_out,
        grad_in,
        size);
}

__global__ void relu_backward_kernel(
    const float *input,
    const float *grad_out,
    float *grad_in,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float grad = input[idx] > 0.0f ? grad_out[idx] : 0.0f;
        atomicAdd(&grad_in[idx], grad);
    }
}

void launch_relu_backward(
    const float *input,
    const float *grad_out,
    float *grad_in,
    int size)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(input, grad_out, grad_in, size);
}

__global__ void softmax_backward_kernel(
    const float *softmax_output,
    const float *grad_out,
    float *grad_in,
    int outer_size,
    int softmax_size,
    int inner_size)
{
    /*
      Softmax backward.

      Formula:
        grad_x_i = y_i * (grad_out_i - sum_j(grad_out_j * y_j))

      where:
        y = softmax(x)

      This computes the Jacobian-vector product efficiently
      without constructing the full Jacobian matrix.
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
      dot = sum_j grad_out_j * softmax_j
    */
    float local_dot = 0.0f;

    for (int j = threadIdx.x; j < softmax_size; j += blockDim.x)
    {
        int idx = base + j * inner_size;
        local_dot += grad_out[idx] * softmax_output[idx];
    }

    __shared__ float shared_dot[256];
    shared_dot[threadIdx.x] = local_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_dot[threadIdx.x] += shared_dot[threadIdx.x + stride];
        }

        __syncthreads();
    }

    float dot = shared_dot[0];

    /*
      Step 2:
      grad_x_i = y_i * (grad_out_i - dot)

      Use atomicAdd because MiniPyPy autograd accumulates gradients.
    */
    for (int j = threadIdx.x; j < softmax_size; j += blockDim.x)
    {
        int idx = base + j * inner_size;
        float grad = softmax_output[idx] * (grad_out[idx] - dot);

        atomicAdd(&grad_in[idx], grad);
    }
}

void launch_softmax_backward(
    const float *softmax_output,
    const float *grad_out,
    float *grad_in,
    int outer_size,
    int softmax_size,
    int inner_size)
{
    int threads = 256;
    int blocks = outer_size * inner_size;

    softmax_backward_kernel<<<blocks, threads>>>(
        softmax_output,
        grad_out,
        grad_in,
        outer_size,
        softmax_size,
        inner_size);
}
