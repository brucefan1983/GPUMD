/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    @author: Hongfu Huang
    @mail: hfhuang@buaa.edu.cn
    This file is part of GNEP.
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

#include "utilities/error.cuh"
#include <cuda_runtime.h>

class DeviceGuard
{
public:
  explicit DeviceGuard(const int device_id)
  {
    CHECK(cudaGetDevice(&previous_device_));
    if (previous_device_ != device_id) {
      CHECK(cudaSetDevice(device_id));
      changed_ = true;
    }
  }

  ~DeviceGuard()
  {
    if (changed_) {
      cudaSetDevice(previous_device_);
    }
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
  int previous_device_ = 0;
  bool changed_ = false;
};
