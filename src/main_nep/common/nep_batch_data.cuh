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
#include "main_nep/dataset.cuh"
#include <memory>
#include <vector>

// Model-owned data indexed in the same way as NEP_Training's datasets.
// Each slot owns its GPU allocations; growing the containers does not copy them.
template <typename T>
class NEP_Batch_Data
{
public:
  T& create(const Dataset& dataset)
  {
    if (dataset.batch_index >= 0 && train.size() <= dataset.batch_index) {
      train.resize(dataset.batch_index + 1);
    }
    auto& devices = dataset.batch_index < 0 ? test : train[dataset.batch_index];
    if (devices.size() <= dataset.device_id) {
      devices.resize(dataset.device_id + 1);
    }
    devices[dataset.device_id].reset(new T);
    return *devices[dataset.device_id];
  }

  T& get(const Dataset& dataset)
  {
    auto& devices = dataset.batch_index < 0 ? test : train[dataset.batch_index];
    return *devices[dataset.device_id];
  }

private:
  std::vector<std::vector<std::unique_ptr<T>>> train;
  std::vector<std::unique_ptr<T>> test;
};
