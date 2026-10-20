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
#include "nep_batch_data.cuh"
#include <string>
#include <utility>

class NEP_Charge_Parameters;

struct NEP_Charge_Labels {
  float charge = 0.0f;
  bool has_bec = false;
  std::vector<float> bec;
};

class NEP_Charge_Dataset
{
public:
  GPU_Vector<float> charge;      // calculated charge in GPU
  GPU_Vector<float> charge_shifted;      // shifted charge in GPU
  GPU_Vector<float> bec;         // Born effective charge in GPU
  std::vector<float> charge_cpu;  // calculated charge in CPU
  std::vector<float> bec_cpu;     // calculated BEC in CPU
  GPU_Vector<float> charge_ref_gpu;       // reference charge in GPU
  GPU_Vector<float> bec_ref_gpu;          // reference BEC in GPU
  std::vector<float> charge_ref_cpu;      // reference charge in CPU
  std::vector<float> bec_ref_cpu;         // reference BEC in CPU

  std::vector<bool> has_bec;

  void construct(const Dataset& dataset, const std::vector<NEP_Charge_Labels>& labels);
  std::vector<float> get_rmse_charge(NEP_Charge_Parameters& para, Dataset& dataset, int device_id);
  std::vector<float> get_rmse_bec(NEP_Charge_Parameters& para, Dataset& dataset, int device_id);
};

void copy_nep_charge_predictions(NEP_Charge_Dataset& data, bool has_bec);
void write_nep_charge_predictions(const std::string& label,
  const std::vector<std::pair<Dataset*, int>>& structures,
  NEP_Batch_Data<NEP_Charge_Dataset>& charge_data, bool has_bec);
