#include "tensor.h"
#include "autograd.h"

#include <stdexcept>
#include <vector>

Tensor Tensor::grad() const
{
  if (!grad_tensor)
  {
    std::vector<float> zeros(size, 0.0f);
    return Tensor(zeros, shape);
  }

  std::vector<float> host_grad = grad_tensor->cpu();
  return Tensor(host_grad, shape);
}

void Tensor::zero_grad()
{
  if (!grad_tensor)
  {
    grad_tensor = new Tensor(shape);
  }

  launch_fill(grad_tensor->d_data, 0.0f, grad_tensor->size);
}

void Tensor::backward()
{
  if (!requires_grad)
    throw std::runtime_error("cannot call backward on tensor that does not require grad");

  // Seed output gradient with 1
  zero_grad();
  launch_fill(grad_tensor->d_data, 1.0f, grad_tensor->size);

  if (grad_fn && grad_fn->op == OpType::MUL)
  {
    Tensor *a = grad_fn->parents[0];
    Tensor *b = grad_fn->parents[1];

    if (a && a->requires_grad)
      a->zero_grad();

    if (b && b->requires_grad && b != a)
      b->zero_grad();

    if (a && a->requires_grad)
    {
      launch_mul_backward(
          grad_tensor->d_data,
          b->d_data,
          a->grad_tensor->d_data,
          a->size);
    }

    if (b && b->requires_grad)
    {
      launch_mul_backward(
          grad_tensor->d_data,
          a->d_data,
          b->grad_tensor->d_data,
          b->size);
    }
  }
  else if (grad_fn && grad_fn->op == OpType::SUM)
  {
    Tensor *a = grad_fn->parents[0];

    if (a && a->requires_grad)
    {
      a->zero_grad();

      launch_sum_backward(
          grad_tensor->d_data,
          a->grad_tensor->d_data,
          a->size);
    }
  }
}
