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
#include "main_nep/common/nep_batch_data.cuh"
#include <string>
#include <utility>

struct TNEP_Atomic_Labels {
  int has_atomic_virial = 0;
  int atomic_virial_diag_only = 0;
  std::vector<float> avirialxx;
  std::vector<float> avirialyy;
  std::vector<float> avirialzz;
  std::vector<float> avirialxy;
  std::vector<float> avirialyz;
  std::vector<float> avirialzx;
};

class TNEP_Atomic_Dataset
{
public:
  GPU_Vector<float> avirial_ref_gpu;
  std::vector<float> avirial_ref_cpu;
  bool atomic_virial_diag_only;

  void construct(const Dataset& dataset, const std::vector<TNEP_Atomic_Labels>& labels);
  std::vector<float> get_rmse(NEP_Parameters& para, Dataset& dataset, bool use_weight, int device_id);
};

void write_tnep_predictions(const std::string& label,
  const std::vector<std::pair<Dataset*, int>>& structures,
  NEP_Batch_Data<TNEP_Atomic_Dataset>& atomic_data, bool dipole, bool atomic);
