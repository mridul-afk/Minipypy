#pragma once

#include <vector>

class Tensor
{
public:
   float *d_data;
   int size;

   std::vector<int> shape;
   std::vector<int> stride;

   bool is_cuda;

   Tensor(int size);
   Tensor(const std::vector<float> &host_data);
   Tensor(std::vector<int> shape);

   // No copying because Tensor owns GPU memory
   Tensor(const Tensor &) = delete;
   Tensor &operator=(const Tensor &) = delete;

   // Move support for returning Tensor from operators
   Tensor(Tensor &&other) noexcept;
   Tensor &operator=(Tensor &&other) noexcept;

   ~Tensor();

   Tensor operator+(const Tensor &other) const;
   Tensor operator*(const Tensor &other) const;
   Tensor operator-(const Tensor &other) const;
   Tensor operator/(const Tensor &other) const;

   std::vector<float> cpu() const;
};
