#pragma once

#include <vector>
#include <memory>

#include "autograd.h"

class Tensor
{
public:
   float *d_data;
   int size;

   std::vector<int> shape;
   std::vector<int> stride;

   bool is_cuda;
   bool requires_grad;

   Tensor *grad_tensor;
   std::shared_ptr<AutogradNode> grad_fn;

   Tensor(int size, bool requires_grad = false);
   Tensor(const std::vector<float> &host_data, bool requires_grad = false);
   Tensor(std::vector<int> shape, bool requires_grad = false);
   Tensor(const std::vector<float> &host_data, std::vector<int> shape, bool requires_grad = false);

   // No copying because Tensor owns GPU memory
   Tensor(const Tensor &) = delete;
   Tensor &operator=(const Tensor &) = delete;

   // Move support for returning Tensor from operators
   Tensor(Tensor &&other) noexcept;
   Tensor &operator=(Tensor &&other) noexcept;

   ~Tensor();

   // OPERATOR OVERLOADING
   Tensor operator+(const Tensor &other) const;
   Tensor operator*(const Tensor &other) const;
   Tensor operator-(const Tensor &other) const;
   Tensor operator/(const Tensor &other) const;

   // TENSOR-SCALAR OPS
   Tensor mul_scalar(float scalar) const;
   Tensor add_scalar(float scalar) const;
   Tensor sub_scalar(float scalar) const;
   Tensor rsub_scalar(float scalar) const;
   Tensor div_scalar(float scalar) const;
   Tensor rdiv_scalar(float scalar) const;

   // TENSOR OPS
   std::vector<float> cpu() const;
   std::vector<int> get_shape() const;
   int ndim() const;
   int numel() const;
   std::vector<int> get_stride() const;

   Tensor matmul(const Tensor &other) const;
   Tensor reshape(std::vector<int> new_shape) const;
   Tensor flatten() const;
   Tensor transpose() const;
   Tensor transpose_last_two_dims() const;
   Tensor sum_to_shape(std::vector<int> target_shape) const;
   Tensor sum() const;
   Tensor mean() const;
   Tensor max() const;
   Tensor clone() const;

   // AUTOGRAD
   Tensor grad() const;
   void zero_grad();
   void backward();
   Tensor detach() const;
   Tensor &requires_grad_(bool value = true);

   // ACTIVATION FUNCTIONS
   Tensor relu() const;
   Tensor softmax(int dim = -1) const;
   Tensor cross_entropy(const Tensor &target) const;
   Tensor bce_with_logits(const Tensor &target) const;

   // SCALAR FUNCTIONS
   Tensor operator+(float scalar) const;
   Tensor operator-(float scalar) const;
   Tensor operator*(float scalar) const;
   Tensor operator/(float scalar) const;

   // MATH FUNCTIONS
   Tensor sqrt() const;
};

Tensor operator+(float scalar, const Tensor &t);
Tensor operator-(float scalar, const Tensor &t);
Tensor operator*(float scalar, const Tensor &t);
Tensor operator/(float scalar, const Tensor &t);
