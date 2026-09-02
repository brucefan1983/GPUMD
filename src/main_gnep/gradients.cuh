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

struct Gradients {
  void allocate(int N, int Nc, int num_variables, int number_of_variables_ann, int dim) {
    grad_sum.resize(
      static_cast<size_t>(Nc) * static_cast<size_t>(num_variables), 0.0f);
    local_sum.resize(num_variables, 0.0f);
    E_wb_grad.resize(
      static_cast<size_t>(N) * static_cast<size_t>(number_of_variables_ann), 0.0f);
    Fp_wb.resize(
      static_cast<size_t>(N) * static_cast<size_t>(number_of_variables_ann) *
        static_cast<size_t>(dim),
      0.0f);
  }
  void clear() {
    E_wb_grad.fill(0.0f);
    grad_sum.fill(0.0f);
    Fp_wb.fill(0.0f);
  }
  GPU_Vector<float> E_wb_grad;      // energy w.r.t. w0, b0, w1, b1
  GPU_Vector<double> grad_sum;
  GPU_Vector<double> local_sum; // deterministic per-device configuration sum
  GPU_Vector<float> Fp_wb;          // gradient of descriptors w.r.t. w0, b0, w1, b1
};
