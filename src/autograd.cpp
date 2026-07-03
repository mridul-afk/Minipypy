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
