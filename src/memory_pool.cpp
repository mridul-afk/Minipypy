#include "memory_pool.h"

#include <cuda_runtime.h>
#include <stdexcept>

CudaMemoryPool &CudaMemoryPool::instance()
{
  static CudaMemoryPool pool;
  return pool;
}

void *CudaMemoryPool::allocate(size_t bytes)
{
  auto &blocks = free_blocks[bytes];

  if (!blocks.empty())
  {
    void *ptr = blocks.back();
    blocks.pop_back();
    return ptr;
  }

  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, bytes);

  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(err));

  return ptr;
}

void CudaMemoryPool::deallocate(void *ptr, size_t bytes)
{
  if (!ptr)
    return;

  free_blocks[bytes].push_back(ptr);
}

CudaMemoryPool::~CudaMemoryPool()
{
  for (auto &pair : free_blocks)
  {
    for (void *ptr : pair.second)
    {
      cudaFree(ptr);
    }
  }
}
