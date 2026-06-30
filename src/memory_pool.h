#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

class CudaMemoryPool
{
private:
  std::unordered_map<size_t, std::vector<void *>> free_blocks;

  CudaMemoryPool() = default;

public:
  CudaMemoryPool(const CudaMemoryPool &) = delete;
  CudaMemoryPool &operator=(const CudaMemoryPool &) = delete;

  static CudaMemoryPool &instance();

  void *allocate(size_t bytes);
  void deallocate(void *ptr, size_t bytes);

  ~CudaMemoryPool();
};
