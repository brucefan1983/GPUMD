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
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class NEP_Parameters;
class Dataset;
class NEP_Neighbor;

struct NEP_ZBL {
  bool enabled = false;
  bool use_typewise_cutoff_zbl = false;
  float typewise_cutoff_zbl_factor = 0.65f;
  bool flexible = false;
  float rc_inner = 1.0f;
  float rc_outer = 2.0f;
  int num_types;
  float para[550];
  int atomic_numbers[NUM_ELEMENTS];
};

void initialize_nep_zbl(const NEP_Parameters& para, NEP_ZBL& zbl);

// GPU operations use the current device selected by the caller.
void update_nep_q_scaler(
  int N, int dim, const float* descriptors,
  float* q_scaler, float* q_scaler_max, float* q_scaler_min);

void output_nep_descriptors(
  const Dataset& dataset,
  int dim,
  int output_mode,
  const std::vector<float>& q_scaler,
  GPU_Vector<float>& descriptors);

// Clear forces and diagonal virials. The radial kernel overwrites off-diagonal virials.
void zero_nep_outputs(Dataset& dataset);

// Clear forces and all six virial components for long-range models.
void zero_nep_long_range_outputs(Dataset& dataset);

void add_nep_zbl(const NEP_ZBL& zbl, Dataset& dataset, const NEP_Neighbor& neighbor);
