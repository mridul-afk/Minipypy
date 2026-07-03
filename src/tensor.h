#pragma once

#include <vector>
#include <memory>

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
   Tensor sum() const;
   Tensor mean() const;
   Tensor max() const;

   Tensor grad() const;
   void zero_grad();
};
