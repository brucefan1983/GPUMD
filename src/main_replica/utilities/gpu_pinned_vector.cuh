/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include <type_traits>

// Page-locked host storage for genuinely asynchronous host-device copies.
template <typename T>
class GPU_Pinned_Vector
{
  static_assert(
    std::is_trivially_copyable<T>::value,
    "GPU_Pinned_Vector requires a trivially copyable element type.");

public:
  GPU_Pinned_Vector() = default;

  explicit GPU_Pinned_Vector(const size_t size) { resize(size); }

  GPU_Pinned_Vector(const size_t size, const T value)
  {
    resize(size);
    fill(value);
  }

  ~GPU_Pinned_Vector() { clear(); }

  GPU_Pinned_Vector(const GPU_Pinned_Vector&) = delete;
  GPU_Pinned_Vector& operator=(const GPU_Pinned_Vector&) = delete;

  GPU_Pinned_Vector(GPU_Pinned_Vector&& other) noexcept
    : size_(other.size_), data_(other.data_)
  {
    other.size_ = 0;
    other.data_ = nullptr;
  }

  GPU_Pinned_Vector& operator=(GPU_Pinned_Vector&& other) noexcept
  {
    if (this != &other) {
      clear();
      size_ = other.size_;
      data_ = other.data_;
      other.size_ = 0;
      other.data_ = nullptr;
    }
    return *this;
  }

  void resize(const size_t size)
  {
    clear();
    if (size > 0)
      CHECK(gpuHostAlloc((void**)&data_, sizeof(T) * size, gpuHostAllocDefault));
    size_ = size;
  }

  void clear()
  {
    if (data_ != nullptr) {
      CHECK(gpuFreeHost(data_));
      data_ = nullptr;
    }
    size_ = 0;
  }

  void fill(const T value)
  {
    for (size_t i = 0; i < size_; ++i)
      data_[i] = value;
  }

  T& operator[](const size_t index) { return data_[index]; }
  const T& operator[](const size_t index) const { return data_[index]; }

  bool empty() const { return size_ == 0; }
  size_t size() const { return size_; }
  T const* data() const { return data_; }
  T* data() { return data_; }

private:
  size_t size_ = 0;
  T* data_ = nullptr;
};
