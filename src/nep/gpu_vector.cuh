/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "error.cuh"

namespace
{
template <typename T>
void __global__ gpu_fill(const size_t size, const T value, T* data)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size)
    data[i] = value;
}
} // anonymous namespace

enum class Memory_Type {
  global = 0, // global memory, also called (linear) device memory
  managed     // managed memory, also called unified memory
};

template <typename T>
class GPU_Vector
{
public:
  // default constructor
  GPU_Vector()
  {
    size_ = 0;
    memory_ = 0;
    memory_type_ = Memory_Type::global;
    allocated_ = false;
  }

  // only allocate memory
  GPU_Vector(const size_t size, const Memory_Type memory_type = Memory_Type::global)
  {
    allocated_ = false;
    resize(size, memory_type);
  }

  // allocate memory and initialize
  GPU_Vector(const size_t size, const T value, const Memory_Type memory_type = Memory_Type::global)
  {
    allocated_ = false;
    resize(size, value, memory_type);
  }

  // deallocate memory
  ~GPU_Vector()
  {
    if (allocated_) {
      CHECK(cudaFree(data_));
      allocated_ = false;
    }
  }

  // only allocate memory
  void resize(const size_t size, const Memory_Type memory_type = Memory_Type::global)
  {
    size_ = size;
    memory_ = size_ * sizeof(T);
    memory_type_ = memory_type;
    if (allocated_) {
      CHECK(cudaFree(data_));
      allocated_ = false;
    }
    if (memory_type_ == Memory_Type::global) {
      CHECK(cudaMalloc((void**)&data_, memory_));
      allocated_ = true;
    } else {
      CHECK(cudaMallocManaged((void**)&data_, memory_));
      allocated_ = true;
    }
  }

  // allocate memory and initialize
  void resize(const size_t size, const T value, const Memory_Type memory_type = Memory_Type::global)
  {
    size_ = size;
    memory_ = size_ * sizeof(T);
    memory_type_ = memory_type;
    if (allocated_) {
      CHECK(cudaFree(data_));
      allocated_ = false;
    }
    if (memory_type == Memory_Type::global) {
      CHECK(cudaMalloc((void**)&data_, memory_));
      allocated_ = true;
    } else {
      CHECK(cudaMallocManaged((void**)&data_, memory_));
      allocated_ = true;
    }
    fill(value);
  }

  // copy data from host with the default size
  void copy_from_host(const T* h_data)
  {
    CHECK(cudaMemcpy(data_, h_data, memory_, cudaMemcpyHostToDevice));
  }

  // copy data from host with a given size
  void copy_from_host(const T* h_data, const size_t size)
  {
    const size_t memory = sizeof(T) * size;
    CHECK(cudaMemcpy(data_, h_data, memory, cudaMemcpyHostToDevice));
  }

  // copy data from device with the default size
  void copy_from_device(const T* d_data)
  {
    CHECK(cudaMemcpy(data_, d_data, memory_, cudaMemcpyDeviceToDevice));
  }

  // copy data from device with a given size
  void copy_from_device(const T* d_data, const size_t size)
  {
    const size_t memory = sizeof(T) * size;
    CHECK(cudaMemcpy(data_, d_data, memory, cudaMemcpyDeviceToDevice));
  }

  // copy data to host with the default size
  void copy_to_host(T* h_data)
  {
    CHECK(cudaMemcpy(h_data, data_, memory_, cudaMemcpyDeviceToHost));
  }

  // copy data to host with a given size
  void copy_to_host(T* h_data, const size_t size)
  {
    const size_t memory = sizeof(T) * size;
    CHECK(cudaMemcpy(h_data, data_, memory, cudaMemcpyDeviceToHost));
  }

  // copy data to device with the default size
  void copy_to_device(T* d_data)
  {
    CHECK(cudaMemcpy(d_data, data_, memory_, cudaMemcpyDeviceToDevice));
  }

  // copy data to device with a given size
  void copy_to_device(T* d_data, const size_t size)
  {
    const size_t memory = sizeof(T) * size;
    CHECK(cudaMemcpy(d_data, data_, memory, cudaMemcpyDeviceToDevice));
  }

  // give "value" to each element
  void fill(const T value)
  {
    const int block_size = 128;
    const int grid_size = (size_ + block_size - 1) / block_size;
    gpu_fill<<<grid_size, block_size>>>(size_, value, data_);
    CUDA_CHECK_KERNEL
  }

  // the [] operator
  T& operator[](int index) { return data_[index]; }

  // some getters
  size_t size() const { return size_; }
  T const* data() const { return data_; }
  T* data() { return data_; }

private:
  bool allocated_;          // true for allocated memory
  size_t size_;             // number of elements
  size_t memory_;           // memory in bytes
  Memory_Type memory_type_; // global or unified memory
  T* data_;                 // data pointer
};
