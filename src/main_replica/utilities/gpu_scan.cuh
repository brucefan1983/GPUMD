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

#include "gpu_vector.cuh"
#ifdef USE_HIP
#include <rocprim/device/device_scan.hpp>
#else
#include <cub/device/device_scan.cuh>
#endif

// Persistent temporary storage avoids an allocation and device synchronization
// for every scan. A workspace may only be used by one stream at a time.
class GPU_Exclusive_Scan
{
public:
  GPU_Exclusive_Scan() = default;
  GPU_Exclusive_Scan(const GPU_Exclusive_Scan&) = delete;
  GPU_Exclusive_Scan& operator=(const GPU_Exclusive_Scan&) = delete;

  void run(
    const int* input,
    int* output,
    const int number_of_items,
    const gpuStream_t stream)
  {
    if (number_of_items <= 0)
      return;

    if (number_of_items > capacity_) {
      size_t required_bytes = 0;
      query_storage_size(input, output, number_of_items, stream, required_bytes);
      if (required_bytes > temporary_storage_.size()) {
        if (temporary_storage_.size() != 0)
          CHECK(gpuStreamSynchronize(stream));
        temporary_storage_.resize(required_bytes);
      }
      capacity_ = number_of_items;
    }

    size_t storage_bytes = temporary_storage_.size();
#ifdef USE_HIP
    CHECK(rocprim::exclusive_scan(
      temporary_storage_.data(),
      storage_bytes,
      input,
      output,
      0,
      static_cast<size_t>(number_of_items),
      rocprim::plus<int>(),
      stream));
#else
    CHECK(cub::DeviceScan::ExclusiveSum(
      temporary_storage_.data(),
      storage_bytes,
      input,
      output,
      number_of_items,
      stream));
#endif
  }

private:
  static void query_storage_size(
    const int* input,
    int* output,
    const int number_of_items,
    const gpuStream_t stream,
    size_t& required_bytes)
  {
#ifdef USE_HIP
    CHECK(rocprim::exclusive_scan(
      nullptr,
      required_bytes,
      input,
      output,
      0,
      static_cast<size_t>(number_of_items),
      rocprim::plus<int>(),
      stream));
#else
    CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr, required_bytes, input, output, number_of_items, stream));
#endif
  }

  int capacity_ = 0;
  GPU_Vector<unsigned char> temporary_storage_;
};
