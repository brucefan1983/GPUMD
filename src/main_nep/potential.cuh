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
#include "utilities/gpu_vector.cuh"
#include <vector>
class Dataset;
class Parameters;

class NEP_Neighbor
{
public:
  GPU_Vector<int> NL_radial;
  GPU_Vector<int> NL_angular;
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;

  void prepare(Parameters& para, Dataset& dataset, int device_id);

private:
  GPU_Vector<int> atomic_numbers;
  GPU_Vector<float> rc_radial;
  GPU_Vector<float> rc_angular;
  const Dataset* neighbor_dataset = nullptr;
};

class Potential
{
public:
  virtual ~Potential() = default;
  virtual void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int DeviceCount) = 0;

protected:
  NEP_Neighbor neighbor[16];
};
