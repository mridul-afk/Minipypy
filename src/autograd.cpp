#include "tensor.h"
#include "autograd.h"

#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <algorithm>

void launch_relu_backward(
    const float *input,
    const float *grad_out,
    float *grad_in,
    int size);

void launch_softmax_backward(
    const float *softmax_output,
    const float *grad_out,
    float *grad_in,
    int outer_size,
    int softmax_size,
    int inner_size);

void launch_cross_entropy_backward(
    const float *logits,
    const float *target,
    const float *grad_out,
    float *grad_logits,
    int batch_size,
    int num_classes);

void launch_bce_with_logits_backward(
    const float *logits,
    const float *target,
    const float *grad_out,
    float *grad_logits,
    int size);

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

    else if (tensor->grad_fn->op == OpType::MEAN)
    {
      Tensor *a = tensor->grad_fn->parents[0];

      if (a && a->requires_grad)
      {
        launch_mean_backward(
            tensor->grad_tensor->d_data,
            a->grad_tensor->d_data,
            a->size);
      }
    }

    else if (tensor->grad_fn->op == OpType::ADD)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a && a->requires_grad)
      {
        Tensor grad_a = tensor->grad_tensor->clone();

        if (grad_a.shape != a->shape)
        {
          grad_a = grad_a.sum_to_shape(a->shape);
        }

        launch_add_backward(
            grad_a.d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        Tensor grad_b = tensor->grad_tensor->clone();

        if (grad_b.shape != b->shape)
        {
          grad_b = grad_b.sum_to_shape(b->shape);
        }

        launch_add_backward(
            grad_b.d_data,
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
        Tensor grad_a = tensor->grad_tensor->clone();

        if (grad_a.shape != a->shape)
        {
          grad_a = grad_a.sum_to_shape(a->shape);
        }

        launch_sub_backward_left(
            grad_a.d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        Tensor grad_b = tensor->grad_tensor->clone();

        if (grad_b.shape != b->shape)
        {
          grad_b = grad_b.sum_to_shape(b->shape);
        }

        launch_sub_backward_right(
            grad_b.d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }

    else if (tensor->grad_fn->op == OpType::MUL)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a && a->requires_grad)
      {
        Tensor grad_a = tensor->grad_tensor->clone() * (*b);

        if (grad_a.shape != a->shape)
        {
          grad_a = grad_a.sum_to_shape(a->shape);
        }

        launch_add_backward(
            grad_a.d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        Tensor grad_b = tensor->grad_tensor->clone() * (*a);

        if (grad_b.shape != b->shape)
        {
          grad_b = grad_b.sum_to_shape(b->shape);
        }

        launch_add_backward(
            grad_b.d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }

    else if (tensor->grad_fn->op == OpType::DIV)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a && a->requires_grad)
      {
        Tensor grad_a = tensor->grad_tensor->clone() / (*b);

        if (grad_a.shape != a->shape)
        {
          grad_a = grad_a.sum_to_shape(a->shape);
        }

        launch_add_backward(
            grad_a.d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        Tensor grad_b = (tensor->grad_tensor->clone() * (*a)) / ((*b) * (*b));

        if (grad_b.shape != b->shape)
        {
          grad_b = grad_b.sum_to_shape(b->shape);
        }

        // Because d(a / b) / db = -a / b^2
        launch_sub_backward_right(
            grad_b.d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }

    else if (tensor->grad_fn->op == OpType::MATMUL)
    {
      Tensor *a = tensor->grad_fn->parents[0];
      Tensor *b = tensor->grad_fn->parents[1];

      if (a->shape.size() < 2 || b->shape.size() < 2)
        throw std::runtime_error("matmul backward requires tensors with at least 2 dimensions");

      if (a && a->requires_grad)
      {
        Tensor b_T = b->transpose_last_two_dims();

        Tensor grad_a = tensor->grad_tensor->matmul(b_T);

        if (grad_a.shape != a->shape)
        {
          grad_a = grad_a.sum_to_shape(a->shape);
        }

        launch_add_backward(
            grad_a.d_data,
            a->grad_tensor->d_data,
            a->size);
      }

      if (b && b->requires_grad)
      {
        Tensor a_T = a->transpose_last_two_dims();

        Tensor grad_b = a_T.matmul(*tensor->grad_tensor);

        if (grad_b.shape != b->shape)
        {
          grad_b = grad_b.sum_to_shape(b->shape);
        }

        launch_add_backward(
            grad_b.d_data,
            b->grad_tensor->d_data,
            b->size);
      }
    }
    else if (tensor->grad_fn->op == OpType::RELU)
    {
      Tensor *a = tensor->grad_fn->parents[0];

      if (a && a->requires_grad)
      {
        launch_relu_backward(
            a->d_data,
            tensor->grad_tensor->d_data,
            a->grad_tensor->d_data,
            a->size);
      }
    }
    else if (tensor->grad_fn->op == OpType::SOFTMAX)
    {
      Tensor *a = tensor->grad_fn->parents[0];

      if (a && a->requires_grad)
      {
        int ndim_count = static_cast<int>(a->shape.size());

        if (ndim_count == 0)
        {
          throw std::runtime_error("softmax backward requires at least 1D tensor");
        }

        int dim = tensor->grad_fn->dim;

        if (dim < 0 || dim >= ndim_count)
        {
          throw std::runtime_error("softmax backward dim out of range");
        }

        int outer_size = 1;
        for (int i = 0; i < dim; i++)
        {
          outer_size *= a->shape[i];
        }

        int softmax_size = a->shape[dim];

        int inner_size = 1;
        for (int i = dim + 1; i < ndim_count; i++)
        {
          inner_size *= a->shape[i];
        }

        if (tensor->grad_fn->saved_tensors.empty())
        {
          throw std::runtime_error("softmax backward missing saved softmax output");
        }

        /*
          Use saved_tensors.back().

          Why?
            add_parent_to_node() may also store intermediate parent tensors
            inside saved_tensors.

          The softmax output is pushed after add_parent_to_node(),
          so it is always the last saved tensor.
        */
        Tensor *softmax_output = tensor->grad_fn->saved_tensors.back().get();

        launch_softmax_backward(
            softmax_output->d_data,
            tensor->grad_tensor->d_data,
            a->grad_tensor->d_data,
            outer_size,
            softmax_size,
            inner_size);
      }
    }
    else if (tensor->grad_fn->op == OpType::CROSS_ENTROPY)
    {
      Tensor *logits = tensor->grad_fn->parents[0];

      if (logits && logits->requires_grad)
      {
        if (logits->shape.size() != 2)
        {
          throw std::runtime_error("cross_entropy backward expects logits with shape [batch, classes]");
        }

        int batch_size = logits->shape[0];
        int num_classes = logits->shape[1];

        if (tensor->grad_fn->saved_tensors.empty())
        {
          throw std::runtime_error("cross_entropy backward missing saved target");
        }

        /*
          The saved target is pushed after add_parent_to_node().
          Use back() because add_parent_to_node() may save intermediate parents.
        */
        Tensor *target = tensor->grad_fn->saved_tensors.back().get();

        launch_cross_entropy_backward(
            logits->d_data,
            target->d_data,
            tensor->grad_tensor->d_data,
            logits->grad_tensor->d_data,
            batch_size,
            num_classes);
      }
    }
    else if (tensor->grad_fn->op == OpType::BCE_WITH_LOGITS)
    {
      Tensor *logits = tensor->grad_fn->parents[0];

      if (logits && logits->requires_grad)
      {
        if (tensor->grad_fn->saved_tensors.empty())
        {
          throw std::runtime_error("bce_with_logits backward missing saved target");
        }

        Tensor *target = tensor->grad_fn->saved_tensors.back().get();

        launch_bce_with_logits_backward(
            logits->d_data,
            target->d_data,
            tensor->grad_tensor->d_data,
            logits->grad_tensor->d_data,
            logits->size);
      }
    }
  }
}
