#pragma once

#include <string>
#include <vector>

class Tensor;

enum class OpType
{
  NONE,
  MUL,
  SUM
};

struct AutogradNode
{
  OpType op;
  std::vector<Tensor *> parents;

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
