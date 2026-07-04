#include "tensor.h"
#include "autograd.h"

#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <algorithm>

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

void build_topo(
    Tensor *tensor,
    std::unordered_set<Tensor *> &visited,
    std::vector<Tensor *> &topo)
{
  if (!tensor)
    return;

  if (visited.count(tensor))
    return;

  visited.insert(tensor);

  if (tensor->grad_fn)
  {
    for (Tensor *parent : tensor->grad_fn->parents)
    {
      build_topo(parent, visited, topo);
    }
  }

  topo.push_back(tensor);
}

void Tensor::backward()
{
  if (!requires_grad)
    throw std::runtime_error("cannot call backward on tensor that does not require grad");

  std::unordered_set<Tensor *> visited;
  std::vector<Tensor *> topo;

  build_topo(this, visited, topo);

  for (Tensor *tensor : topo)
  {
    if (tensor && tensor->requires_grad)
      tensor->zero_grad();
  }

  launch_fill(grad_tensor->d_data, 1.0f, grad_tensor->size);

  std::reverse(topo.begin(), topo.end());

  for (Tensor *tensor : topo)
  {
    if (!tensor || !tensor->grad_fn)
      continue;

    if (tensor->grad_fn->op == OpType::SUM)
    {
      Tensor *a = tensor->grad_fn->parents[0];

      if (a && a->requires_grad)
      {
        launch_sum_backward(
            tensor->grad_tensor->d_data,
            a->grad_tensor->d_data,
            a->size);
      }
    }

    else if (tensor->grad_fn->op == OpType::MUL)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a && a->requires_grad)
      {
        launch_mul_backward(
            tensor->grad_tensor->d_data,
            b->d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        launch_mul_backward(
            tensor->grad_tensor->d_data,
            a->d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }
    else if (tensor->grad_fn->op == OpType::ADD)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a && a->requires_grad)
      {
        launch_add_backward(
            tensor->grad_tensor->d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        launch_add_backward(
            tensor->grad_tensor->d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }
    else if (tensor->grad_fn->op == OpType::SUB)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a && a->requires_grad)
      {
        launch_sub_backward_left(
            tensor->grad_tensor->d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        launch_sub_backward_right(
            tensor->grad_tensor->d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }
  }
}
