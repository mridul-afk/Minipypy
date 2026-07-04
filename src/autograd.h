#pragma once

#include <string>
#include <vector>
#include <memory>

class Tensor;

enum class OpType
{
  NONE,
  MUL,
  SUM,
  ADD,
  SUB
};

struct AutogradNode
{
  OpType op;
  std::vector<Tensor *> parents;

  // Own intermediate tensors so temporaries like (x*x) survive.
  std::vector<std::shared_ptr<Tensor>> saved_tensors;

  AutogradNode()
      : op(OpType::NONE) {}

  AutogradNode(OpType op, std::vector<Tensor *> parents)
      : op(op), parents(parents) {}
};

void launch_fill(float *data, float value, int size);

void launch_mul_backward(
    const float *grad_out,
    const float *other,
    float *grad_in,
    int size);

void launch_sum_backward(
    const float *grad_out,
    float *grad_in,
    int size);

void launch_add_backward(
    const float *grad_out,
    float *grad_in,
    int size);

void launch_sub_backward_left(
    const float *grad_out,
    float *grad_in,
    int size);

void launch_sub_backward_right(
    const float *grad_out,
    float *grad_in,
    int size);
