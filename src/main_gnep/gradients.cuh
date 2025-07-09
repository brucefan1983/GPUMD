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

static void print_memory_info(const char* stage) {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    float free_gb = free_memory / (1024.0 * 1024.0 * 1024.0);
    float total_gb = total_memory / (1024.0 * 1024.0 * 1024.0);
    float used_gb = total_gb - free_gb;
    
    printf("%s:\n", stage);
    printf("  Free memory:  %.2f GB\n", free_gb);
    printf("  Used memory:  %.2f GB\n", used_gb);
    printf("  Total memory: %.2f GB\n", total_gb);
}

struct Gradients {
  void resize(int N, int num_variables, int number_of_variables_ann, int dim) {
    grad_sum.resize(num_variables, 0.0f);
    E_wb_grad.resize(N * number_of_variables_ann, 0.0f);
    Fp_wb.resize(N * number_of_variables_ann * dim, 0.0f);
  }
  void clear() {
    E_wb_grad.fill(0.0f);
    grad_sum.fill(0.0f);
    Fp_wb.fill(0.0f);
  }
  GPU_Vector<float> E_wb_grad;      // energy w.r.t. w0, b0, w1, b1
  GPU_Vector<float> grad_sum;
  GPU_Vector<float> Fp_wb;          // gradient of descriptors w.r.t. w0, b0, w1, b1
};
