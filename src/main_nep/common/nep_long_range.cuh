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
#include "utilities/gpu_vector.cuh"

class Dataset;
class NEP_Charge_Dataset;

// All GPU operations use the device selected by the caller.
// K-point generation, storage, and cache lifetime remain model-specific.
void enforce_nep_total_charge(Dataset& dataset, NEP_Charge_Dataset& charge_dataset);

void subtract_nep_charge_derivative_mean(const Dataset& dataset, float* D_real);

void initialize_nep_bec(Dataset& dataset, NEP_Charge_Dataset& charge_dataset);

void scale_nep_bec(Dataset& dataset, NEP_Charge_Dataset& charge_dataset, const float* sqrt_epsilon_inf);

void compute_nep_structure_factor(
  const Dataset& dataset,
  const float* weights,
  const int* kpoint_offset,
  const float* kx, const float* ky, const float* kz,
  float* S_real, float* S_imag);

void add_nep_vdw_long_range(
  Dataset& dataset,
  float alpha,
  const float* C6,
  const int* kpoint_offset,
  const float* kx, const float* ky, const float* kz,
  const float* G_vdw, const float* G_vdw_virial,
  const float* S_real, const float* S_imag,
  float* D_C6);

// Replace Nc counts with Nc + 1 prefix offsets and return the total.
int update_nep_kpoint_offsets(int Nc, GPU_Vector<int>& offsets);
