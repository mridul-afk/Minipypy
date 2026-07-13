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
    SUB,
    DIV,
    MEAN,
    MATMUL,
    RELU,
    SOFTMAX,
    CROSS_ENTROPY
};

struct AutogradNode
{
    OpType op = OpType::NONE;

    std::vector<Tensor *> parents;

    /*
      saved_tensors keeps temporary/intermediate tensors alive
      until backward runs.
    */
    std::vector<std::shared_ptr<Tensor>> saved_tensors;

    /*
      Some ops need extra metadata for backward.

      Example:
        softmax(dim=1)

      Backward must know which dimension softmax was applied over.
    */
    int dim = -1;
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

void launch_div_backward_left(
    const float *grad_out,
    const float *b,
    float *grad_a,
    int size);

void launch_div_backward_right(
    const float *grad_out,
    const float *a,
    const float *b,
    float *grad_b,
    int size);

void launch_mean_backward(
    const float *grad_out,
    float *grad_in,
    int size);
