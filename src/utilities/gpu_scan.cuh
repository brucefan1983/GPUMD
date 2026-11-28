/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "gpu_vector.cuh"

// Persistent temporary storage avoids an allocation and device synchronization
// for every scan. A workspace may only be used by one stream at a time.
class GPU_Exclusive_Scan
{
public:
  GPU_Exclusive_Scan() = default;
  GPU_Exclusive_Scan(const GPU_Exclusive_Scan&) = delete;
  GPU_Exclusive_Scan& operator=(const GPU_Exclusive_Scan&) = delete;

  void run(const int* input, int* output, const int number_of_items, const gpuStream_t stream);

private:
  int capacity_ = 0;
  GPU_Vector<unsigned char> temporary_storage_;
};
