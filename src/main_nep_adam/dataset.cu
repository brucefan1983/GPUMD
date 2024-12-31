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

#include "dataset.cuh"
#include "mic.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/least_square.cuh"
#include <algorithm> 

void Dataset::copy_structures(std::vector<Structure>& structures_input, int n1, int n2)
{
  Nc = n2 - n1;
  structures.resize(Nc);

  for (int n = 0; n < Nc; ++n) {
    int n_input = n + n1;
    structures[n].num_atom = structures_input[n_input].num_atom;
    structures[n].weight = structures_input[n_input].weight;
    structures[n].has_virial = structures_input[n_input].has_virial;
    structures[n].energy = structures_input[n_input].energy;
    structures[n].has_temperature = structures_input[n_input].has_temperature;
    structures[n].temperature = structures_input[n_input].temperature;
    structures[n].volume = structures_input[n_input].volume;
    for (int k = 0; k < 6; ++k) {
      structures[n].virial[k] = structures_input[n_input].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures[n].box[k] = structures_input[n_input].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      structures[n].box_original[k] = structures_input[n_input].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      structures[n].num_cell[k] = structures_input[n_input].num_cell[k];
    }

    structures[n].type.resize(structures[n].num_atom);
    structures[n].x.resize(structures[n].num_atom);
    structures[n].y.resize(structures[n].num_atom);
    structures[n].z.resize(structures[n].num_atom);
    structures[n].fx.resize(structures[n].num_atom);
    structures[n].fy.resize(structures[n].num_atom);
    structures[n].fz.resize(structures[n].num_atom);

    for (int na = 0; na < structures[n].num_atom; ++na) {
      structures[n].type[na] = structures_input[n_input].type[na];
      structures[n].x[na] = structures_input[n_input].x[na];
      structures[n].y[na] = structures_input[n_input].y[na];
      structures[n].z[na] = structures_input[n_input].z[na];
      structures[n].fx[na] = structures_input[n_input].fx[na];
      structures[n].fy[na] = structures_input[n_input].fy[na];
      structures[n].fz[na] = structures_input[n_input].fz[na];
    }
  }
}

bool Dataset::find_has_type(Parameters& para)
{
  has_type.resize((para.num_types + 1) * Nc, false);
  for (int n = 0; n < Nc; ++n) {
    has_type[para.num_types * Nc + n] = true;
    for (int na = 0; na < structures[n].num_atom; ++na) {
      has_type[structures[n].type[na] * Nc + n] = true;
    }
  }

  // Verify if each type in num_types has at least one true value
  bool all_types_present = true;
  for (int t = 0; t < para.num_types; ++t) {
    bool type_present = false;
    for (int n = 0; n < Nc; ++n) {
      if (has_type[t * Nc + n]) {
        type_present = true;
        break;
      }
    }
    if (!type_present) {
      all_types_present = false;
      break;
    }
  }

  if (!all_types_present) {
    printf("Not all types are present in train set.\n");
  } 
  return all_types_present;
}

void Dataset::find_Na(Parameters& para)
{
  Na_cpu.resize(Nc);
  Na_sum_cpu.resize(Nc);

  N = 0;
  max_Na = 0;
  int num_virial_configurations = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    Na_cpu[nc] = structures[nc].num_atom;
    Na_sum_cpu[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    N += structures[nc].num_atom;
    if (structures[nc].num_atom > max_Na) {
      max_Na = structures[nc].num_atom;
    }
    num_virial_configurations += structures[nc].has_virial;
  }

  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum_cpu[nc] = Na_sum_cpu[nc - 1] + Na_cpu[nc - 1];
  }

  printf("Total number of atoms = %d.\n", N);
  printf("Number of atoms in the largest configuration = %d.\n", max_Na);
  if (para.train_mode == 0 || para.train_mode == 3) {
    printf("Number of configurations having virial = %d.\n", num_virial_configurations);
  }

  Na.resize(Nc);
  Na_sum.resize(Nc);
  Na.copy_from_host(Na_cpu.data());
  Na_sum.copy_from_host(Na_sum_cpu.data());
}

void Dataset::initialize_gpu_data(Parameters& para)
{
  std::vector<double> box_cpu(Nc * 18);
  std::vector<double> box_original_cpu(Nc * 9);
  std::vector<int> num_cell_cpu(Nc * 3);
  std::vector<double> r_cpu(N * 3);
  std::vector<int> type_cpu(N);
  std::vector<int> type_sum_cpu(para.num_types);

  energy.resize(N);
  virial.resize(N * 6);
  force.resize(N * 3);
  energy_cpu.resize(N);
  virial_cpu.resize(N * 6);
  force_cpu.resize(N * 3);

  weight_cpu.resize(Nc);
  energy_ref_cpu.resize(Nc);
  virial_ref_cpu.resize(Nc * 6);
  force_ref_cpu.resize(N * 3);
  temperature_ref_cpu.resize(N);

  for (int n = 0; n < Nc; ++n) {
    weight_cpu[n] = structures[n].weight;
    energy_ref_cpu[n] = structures[n].energy;
    sum_energy_ref += structures[n].energy; 
    for (int k = 0; k < 6; ++k) {
      virial_ref_cpu[k * Nc + n] = structures[n].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      box_cpu[k + n * 18] = structures[n].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      box_original_cpu[k + n * 9] = structures[n].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      num_cell_cpu[k + n * 3] = structures[n].num_cell[k];
    }
    for (int na = 0; na < structures[n].num_atom; ++na) {
      int init_type = structures[n].type[na];
      type_cpu[Na_sum_cpu[n] + na] = init_type;
      type_sum_cpu[init_type]++;
      r_cpu[Na_sum_cpu[n] + na] = structures[n].x[na];
      r_cpu[Na_sum_cpu[n] + na + N] = structures[n].y[na];
      r_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].z[na];
      force_ref_cpu[Na_sum_cpu[n] + na] = structures[n].fx[na];
      force_ref_cpu[Na_sum_cpu[n] + na + N] = structures[n].fy[na];
      force_ref_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].fz[na];
      temperature_ref_cpu[Na_sum_cpu[n] + na] = structures[n].temperature;
    }
  }

  type_weight_gpu.resize(NUM_ELEMENTS);
  energy_ref_gpu.resize(Nc);
  virial_ref_gpu.resize(Nc * 6);
  force_ref_gpu.resize(N * 3);
  temperature_ref_gpu.resize(N);
  weight_gpu.resize(Nc);
  weight_gpu.copy_from_host(weight_cpu.data());
  type_weight_gpu.copy_from_host(para.type_weight_cpu.data());
  energy_ref_gpu.copy_from_host(energy_ref_cpu.data());
  virial_ref_gpu.copy_from_host(virial_ref_cpu.data());
  force_ref_gpu.copy_from_host(force_ref_cpu.data());
  temperature_ref_gpu.copy_from_host(temperature_ref_cpu.data());


  box.resize(Nc * 18);
  box_original.resize(Nc * 9);
  num_cell.resize(Nc * 3);
  r.resize(N * 3);
  type.resize(N);
  type_sum.resize(para.num_types);
  for (int n = 0; n < para.num_types; ++n) {
    printf("Number of atoms type %d = %d\n", n, type_sum_cpu[n]);
  }
  box.copy_from_host(box_cpu.data());
  box_original.copy_from_host(box_original_cpu.data());
  num_cell.copy_from_host(num_cell_cpu.data());
  r.copy_from_host(r_cpu.data());
  type.copy_from_host(type_cpu.data());
  type_sum.copy_from_host(type_sum_cpu.data());
}

void Dataset::initialize_gradients_temp(Parameters& para)
{
  gradients.resize(para.number_of_variables_descriptor, N, para.number_of_variables_ann);
}

static __global__ void gpu_find_neighbor_number(
  const int N,
  const int* Na,
  const int* Na_sum,
  const bool use_typewise_cutoff,
  const double typewise_cutoff_radial_factor,
  const double typewise_cutoff_angular_factor,
  const int* g_type,
  const int* g_atomic_numbers,
  const double g_rc_radial,
  const double g_rc_angular,
  const double* __restrict__ g_box,
  const double* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const double* x,
  const double* y,
  const double* z,
  int* NN_radial,
  int* NN_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const double* __restrict__ box = g_box + 18 * blockIdx.x;
    const double* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            double delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            double delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            double delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            double x12 = x[n2] + delta_x - x1;
            double y12 = y[n2] + delta_y - y1;
            double z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            double rc_radial = g_rc_radial;
            double rc_angular = g_rc_angular;
            if (use_typewise_cutoff) {
              int z1 = g_atomic_numbers[t1];
              int z2 = g_atomic_numbers[t2];
              rc_radial = min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * typewise_cutoff_radial_factor, rc_radial);
              rc_angular = min((COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * typewise_cutoff_angular_factor, rc_angular);
            }
            if (distance_square < rc_radial * rc_radial) {
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              count_angular++;
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

void Dataset::find_neighbor(Parameters& para)
{
  GPU_Vector<int> NN_radial_gpu(N);
  GPU_Vector<int> NN_angular_gpu(N);
  std::vector<int> NN_radial_cpu(N);
  std::vector<int> NN_angular_cpu(N);

  std::vector<int> atomic_numbers_from_zero(para.atomic_numbers.size());
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    atomic_numbers_from_zero[n] = para.atomic_numbers[n] - 1;
  }
  GPU_Vector<int> atomic_numbers(para.atomic_numbers.size());
  atomic_numbers.copy_from_host(atomic_numbers_from_zero.data());

  gpu_find_neighbor_number<<<Nc, 256>>>(
    N,
    Na.data(),
    Na_sum.data(),
    para.use_typewise_cutoff,
    para.typewise_cutoff_radial_factor,
    para.typewise_cutoff_angular_factor,
    type.data(),
    atomic_numbers.data(),
    para.rc_radial,
    para.rc_angular,
    box.data(),
    box_original.data(),
    num_cell.data(),
    r.data(),
    r.data() + N,
    r.data() + N * 2,
    NN_radial_gpu.data(),
    NN_angular_gpu.data());
  CUDA_CHECK_KERNEL

  NN_radial_gpu.copy_to_host(NN_radial_cpu.data());
  NN_angular_gpu.copy_to_host(NN_angular_cpu.data());

  int min_NN_radial = 10000;
  max_NN_radial = -1;
  for (int n = 0; n < N; ++n) {
    if (NN_radial_cpu[n] < min_NN_radial) {
      min_NN_radial = NN_radial_cpu[n];
    }
    if (NN_radial_cpu[n] > max_NN_radial) {
      max_NN_radial = NN_radial_cpu[n];
    }
  }
  int min_NN_angular = 10000;
  max_NN_angular = -1;
  for (int n = 0; n < N; ++n) {
    if (NN_angular_cpu[n] < min_NN_angular) {
      min_NN_angular = NN_angular_cpu[n];
    }
    if (NN_angular_cpu[n] > max_NN_angular) {
      max_NN_angular = NN_angular_cpu[n];
    }
  }

  printf("Radial descriptor with a cutoff of %g A:\n", para.rc_radial);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_radial);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_radial);
  printf("Angular descriptor with a cutoff of %g A:\n", para.rc_angular);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_angular);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_angular);
}

void Dataset::construct(
  Parameters& para, std::vector<Structure>& structures_input, bool require_grad, int n1, int n2, int device_id)
{
  CHECK(cudaSetDevice(device_id));
  copy_structures(structures_input, n1, n2);
  all_type = find_has_type(para);

  error_cpu.resize(Nc);
  error_gpu.resize(Nc);
  diff_gpu.resize(Nc);
  diff_gpu_virial.resize(Nc * 6);

  find_Na(para);
  initialize_gpu_data(para);
  if (require_grad) {
    initialize_gradients_temp(para);
  }
  if (para.calculate_energy_shift && all_type) {
    computeEnergyPerTypeReg(para.num_types,
                            type_sum.data(),
                            sum_energy_ref,
                            0.001,
                            para.energy_shift_gpu.data());
    para.calculate_energy_shift = false;
    std::vector<double> energy_per_type_host(para.num_types);
    CHECK(cudaMemcpy(energy_per_type_host.data(), para.energy_shift_gpu.data(),
                    sizeof(double) * para.num_types, cudaMemcpyDeviceToHost));
    for (int i = 0; i < para.num_types; ++i) {
      printf("energy_per_type_host[%d] = %f\n", i, energy_per_type_host[i]);
    }
  }
  find_neighbor(para);
}

static __global__ void gpu_sum_force_error(
  bool use_weight,
  double force_delta,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  double* g_type_weight,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_fx_ref,
  double* g_fy_ref,
  double* g_fz_ref,
  double* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ double s_error[];
  s_error[tid] = 0.0;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    double fx_ref = g_fx_ref[n];
    double fy_ref = g_fy_ref[n];
    double fz_ref = g_fz_ref[n];
    double dx = g_fx[n] - fx_ref;
    double dy = g_fy[n] - fy_ref;
    double dz = g_fz[n] - fz_ref;
    double diff_square = dx * dx + dy * dy + dz * dz;
    if (use_weight) {
      double type_weight = g_type_weight[g_type[n]];
      diff_square *= type_weight * type_weight;
    }
    if (use_weight && force_delta > 0.0f) {
      double force_magnitude = sqrt(fx_ref * fx_ref + fy_ref * fy_ref + fz_ref * fz_ref);
      diff_square *= force_delta / (force_delta + force_magnitude);
    }
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

static __global__ void sum_gradients_f(
  const int var_per_block,    // 每个block处理的变量数量
  int num_sub_blocks,         // 子block数量，变量分布在多个子block中
  int N,                      // 原子总数
  int Nc,                     // 结构index (batch size)
  bool use_weight,
  int num_var,                  // 每个批次的变量数量
  const double lambda_f,       // 每个结构的原子力差的权重
  int* g_Na,                  // 每个结构的原子数，大小为Nc
  int* g_Na_sum,             // 每个结构在全局原子数组中的原子起始索引，大小为Nc
  int* g_type,               // 原子类型，大小为N
  double* g_type_weight,      // 原子类型权重，大小为num_types + 1
  double force_delta,         // 力差的权重
  const double* __restrict__ g_fx,      // 预测的力
  const double* __restrict__ g_fy,
  const double* __restrict__ g_fz,
  const double* __restrict__ g_fx_ref,  // 参考力
  const double* __restrict__ g_fy_ref,
  const double* __restrict__ g_fz_ref,
  const double* __restrict__ g_weight,  // 结构权重
  const double* __restrict__ g_F_x_grad,   
  const double* __restrict__ g_F_y_grad,
  const double* __restrict__ g_F_z_grad,
  double* __restrict__ g_gradients_sum,  // 累积的梯度
  bool is_descriptor
)
{
  int sub_block_idx = blockIdx.y;
  if (sub_block_idx >= num_sub_blocks) return;

  int start_var = sub_block_idx * var_per_block;
  int end_var = min(start_var + var_per_block, num_var);
  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (bid >= Nc) return;

  int Na = g_Na[bid];      
  int N1 = g_Na_sum[bid];  
  int N2 = N1 + Na;        
  double weight = g_weight[bid];
  const double per_Nc = 2.0 * lambda_f / Na / 3 / Nc;

  for (int c_idx = start_var + tid; c_idx < end_var; c_idx += blockDim.x) {
    if (c_idx >= num_var) continue;
    
    double grad_sum = 0.0;
    
    for (int n = N1; n < N2; ++n) {
      double fx_ref = g_fx_ref[n];
      double fy_ref = g_fy_ref[n];
      double fz_ref = g_fz_ref[n];
      double dx = g_fx[n] - fx_ref;
      double dy = g_fy[n] - fy_ref;
      double dz = g_fz[n] - fz_ref;
      
      if (use_weight) {
        double type_weight = g_type_weight[g_type[n]];
        if (force_delta > 0.0f) {
          double force_magnitude = sqrt(fx_ref * fx_ref + fy_ref * fy_ref + fz_ref * fz_ref);
          type_weight *= sqrt(force_delta / (force_delta + force_magnitude));
        }
        dx *= type_weight;
        dy *= type_weight;
        dz *= type_weight;
      }
      int index = is_descriptor ? c_idx * N + n : num_var * n + c_idx;
      grad_sum += (g_F_x_grad[index] * dx +
                   g_F_y_grad[index] * dy +
                   g_F_z_grad[index] * dz);
    }
    
    atomicAdd(&g_gradients_sum[c_idx], grad_sum * weight * per_Nc);
  }
}

std::vector<double> Dataset::get_rmse_force(Parameters& para, const bool use_weight, const bool require_grad, int device_id)
{
  CHECK(cudaSetDevice(device_id));
  const int block_size = 256;
  gpu_sum_force_error<<<Nc, block_size, sizeof(double) * block_size>>>(
    use_weight,
    para.force_delta,
    Na.data(),
    Na_sum.data(),
    type.data(),
    type_weight_gpu.data(),
    force.data(),
    force.data() + N,
    force.data() + N * 2,
    force_ref_gpu.data(),
    force_ref_gpu.data() + N,
    force_ref_gpu.data() + N * 2,
    error_gpu.data());
  int mem = sizeof(double) * Nc;
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));

  std::vector<double> rmse_array(para.num_types + 1, 0.0);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    double rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += Na_cpu[n];
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      // rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 3));
      rmse_array[t] = rmse_array[t] / (count_array[t] * 3);
    }
  }
  if (require_grad) {
    int num_sub_blocks = (para.number_of_variables_descriptor + block_size - 1) / block_size;
    dim3 grid_dim(Nc, num_sub_blocks);
    sum_gradients_f<<<grid_dim, block_size>>>(
      block_size, num_sub_blocks, N, Nc, 
      use_weight,
      para.number_of_variables_descriptor, 
      para.lambda_f, 
      Na.data(), 
      Na_sum.data(), 
      type.data(), 
      type_weight_gpu.data(), 
      para.force_delta,
      force.data(), 
      force.data() + N,
      force.data() + N * 2,
      force_ref_gpu.data(), 
      force_ref_gpu.data() + N, 
      force_ref_gpu.data() + N * 2, 
      weight_gpu.data(), 
      gradients.F_c_x.data(), 
      gradients.F_c_y.data(), 
      gradients.F_c_z.data(), 
      gradients.grad_c_sum.data(),
      true);
    CUDA_CHECK_KERNEL

    int num_sub_blocks_wb = (para.number_of_variables_ann + block_size - 1) / block_size;
    dim3 grid_dim_wb(Nc, num_sub_blocks_wb);
    sum_gradients_f<<<grid_dim_wb, block_size>>>(
      block_size, num_sub_blocks_wb, N, Nc, 
      use_weight,
      para.number_of_variables_ann, 
      para.lambda_f, 
      Na.data(), 
      Na_sum.data(), 
      type.data(), 
      type_weight_gpu.data(), 
      para.force_delta,
      force.data(), 
      force.data() + N,
      force.data() + N * 2,
      force_ref_gpu.data(), 
      force_ref_gpu.data() + N, 
      force_ref_gpu.data() + N * 2, 
      weight_gpu.data(), 
      gradients.F_wb_grad_x.data(), 
      gradients.F_wb_grad_y.data(), 
      gradients.F_wb_grad_z.data(), 
      gradients.grad_wb_sum.data(),
      false);
    CUDA_CHECK_KERNEL
  }
  return rmse_array;
}

static __global__ void gpu_sum_pe_error(
  int* g_Na, int* g_Na_sum, double* g_pe, double* g_pe_ref, double* diff_gpu, double* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];   // 当前结构的原子数
  int N1 = g_Na_sum[bid]; // 当前结构在全局原子数组中的原子起始索引
  int N2 = N1 + Na;      // 当前结构在全局原子数组中的原子结束索引（不包括）
  extern __shared__ double s_pe[];
  s_pe[tid] = 0.0;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    double diff = (s_pe[0] - g_pe_ref[bid]) / Na;
    diff_gpu[bid] = diff;
    error_gpu[bid] = diff * diff;
  }
}

static __global__ void sum_gradients_e(
  const int var_per_block,
  int num_sub_blocks,
  int N,              // 原子总数
  int Nc,             // 结构index (batch size)
  int num_var,          // 每个批次的变量数量
  const double lambda_e, // 每个结构的原子能量差的权重
  int* g_Na,          // 每个结构的原子数，大小为Nc
  int* g_Na_sum,      // 每个结构在全局原子数组中的原子起始索引，大小为Nc
  const double* __restrict__ g_diff, // 每个结构的原子能量差，大小为Nc
  const double* __restrict__ g_weight, // 每个结构的权重，大小为Nc
  const double* __restrict__ g_E_grad,  // 每个结构的能量的梯度，大小为N * num_c (N * num_ann)
  double* __restrict__ g_gradients_sum,      // 每个结构的梯度之和，大小为num_c (num_ann)
  bool is_descriptor
)
{
  int sub_block_idx = blockIdx.y;
  if (sub_block_idx >= num_sub_blocks) return;

  int start_var = sub_block_idx * var_per_block;
  int end_var = min(start_var + var_per_block, num_var);
  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (bid >= Nc) return;

  int Na = g_Na[bid];   // 当前结构的原子数
  int N1 = g_Na_sum[bid]; // 当前结构在全局原子数组中的原子起始索引
  int N2 = N1 + Na;      // 当前结构在全局原子数组中的原子结束索引（不包括）
  double weight_diff = g_diff[bid] * g_weight[bid];
  const double per_Nc = 2.0 * lambda_e / Nc;

  for (int c_idx = start_var + tid; c_idx < end_var; c_idx += blockDim.x) {
    if (c_idx >= num_var) continue;
    double temp = 0.0;
    for (int n = N1; n < N2; ++n) {
      int index = is_descriptor ? c_idx * N + n : num_var * n + c_idx;
      temp += g_E_grad[index];
    }
    temp *= per_Nc * weight_diff;
    atomicAdd(&g_gradients_sum[c_idx], temp);
  }
}

std::vector<double> Dataset::get_rmse_energy(
  Parameters& para,
  const bool use_weight,
  const bool require_grad,
  int device_id)
{
  CHECK(cudaSetDevice(device_id));

  const int block_size = 256;
  int mem = sizeof(double) * Nc;

  /*
  Nc = 3
  Na = [5, 5, 5]
  Na_sum = [0, 5, 10]
  gridDim.x = grid_size = 3
  blockDim.x = block_size = 256
  blockIdx.x = 0, 1, 2
  threadIdx.x = 0, 1, 2, ..., 255
  */
  gpu_sum_pe_error<<<Nc, block_size, sizeof(double) * block_size>>>(
    Na.data(),
    Na_sum.data(),
    energy.data(),
    energy_ref_gpu.data(),
    diff_gpu.data(),
    error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));

  std::vector<double> rmse_array(para.num_types + 1, 0.0);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    double rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        ++count_array[t];
      }
    }
  }
  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      // rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
      rmse_array[t] = rmse_array[t] / count_array[t];
    }
  }

  if (require_grad) {
    int num_sub_blocks = (para.number_of_variables_descriptor + block_size - 1) / block_size;
    dim3 grid_dim(Nc, num_sub_blocks);
    sum_gradients_e<<<grid_dim, block_size>>>(
      block_size, num_sub_blocks, N, Nc, para.number_of_variables_descriptor, para.lambda_e, 
      Na.data(), Na_sum.data(), diff_gpu.data(), weight_gpu.data(), gradients.E_c.data(), gradients.grad_c_sum.data(),
      true);
    CUDA_CHECK_KERNEL
  //   std::vector<double>grad_c_sum(para.number_of_variables_descriptor);
  //  CHECK(cudaMemcpy(grad_c_sum.data(), gradients.grad_c_sum.data(), para.number_of_variables_descriptor * sizeof(double), cudaMemcpyDeviceToHost));
  //  for (int j = 0; j < para.number_of_variables_descriptor; ++j) {
  //     printf("grad_c_sum[%d] = %f\n", j, grad_c_sum[j]);
  //   }

    int num_sub_blocks_wb = (para.number_of_variables_ann + block_size - 1) / block_size;
    dim3 grid_dim_wb(Nc, num_sub_blocks_wb);
    sum_gradients_e<<<grid_dim_wb, block_size>>>(
      block_size, num_sub_blocks_wb, N, Nc, para.number_of_variables_ann, para.lambda_e, 
      Na.data(), Na_sum.data(), diff_gpu.data(), weight_gpu.data(), gradients.E_wb_grad.data(), gradients.grad_wb_sum.data(),
      false);
    CUDA_CHECK_KERNEL
    // std::vector<double>grad_wb_sum(para.number_of_variables_ann);
    // CHECK(cudaMemcpy(grad_wb_sum.data(), gradients.grad_wb_sum.data(), para.number_of_variables_ann * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int j = 0; j < para.number_of_variables_ann; ++j) {
    //     printf("grad_wb_sum[%d] = %f\n", j, grad_wb_sum[j]);
    //   }
  }
  return rmse_array;
}

static __global__ void gpu_sum_virial_error(
  const int N,
  const double shear_weight,
  int* g_Na,
  int* g_Na_sum,
  double* g_virial,
  double* g_virial_ref,
  double* diff_gpu,
  double* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ double s_virial[];
  for (int d = 0; d < 6; ++d) {
    s_virial[d * blockDim.x + tid] = 0.0; //size of s_virial is 6 * blockDim.x
}                     // sum of atomic contributions to virial tensor, respectively for xx, yy, zz, xy, yz, zx

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    for (int d = 0; d < 6; ++d) {
      s_virial[d * blockDim.x + tid] += g_virial[d * N + n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncwarp();
  }

  if (tid == 0) {
    double error_sum = 0.0;
    for (int d = 0; d < 6; ++d) {
      double diff = (s_virial[d * blockDim.x + 0] - g_virial_ref[d * gridDim.x + bid]) / Na;
      // printf("s_virial[%d * %d + 0] = %f, g_virial_ref[%d * %d + %d] = %f, diff^2 = %f\n", d, blockDim.x, s_virial[d * blockDim.x + 0], d, gridDim.x, bid, g_virial_ref[d * gridDim.x + bid], diff * diff);
      error_sum += (d >= 3) ? (shear_weight * diff * diff) : (diff * diff);
      diff_gpu[bid * 6 + d] = (d >= 3) ? shear_weight * diff : diff;
      // diff_gpu[d * gridDim.x + bid] = (d >= 3) ? shear_weight * diff : diff;
    }
    error_gpu[bid] = error_sum;
  }
}

static __global__ void sum_gradients_v(
  const int var_per_block,
  int num_sub_blocks,
  int N,              
  int Nc,             
  int count_Nc,      
  int num_var,      
  const double lambda_v,
  int* g_Na,          
  int* g_Na_sum,   
  const double* __restrict__ g_diff,    // 每个结构的六个virial差，大小为Nc * 6
  const double* __restrict__ g_weight,  // 每个结构的权重，大小为Nc
  const double* __restrict__ g_V_xx,  // V_c_xx梯度，大小为N * num_c (N * num_ann)
  const double* __restrict__ g_V_yy,  // V_c_yy梯度，大小为N * num_c (N * num_ann)
  const double* __restrict__ g_V_zz,  // V_c_zz梯度，大小为N * num_c (N * num_ann)
  const double* __restrict__ g_V_xy,  // V_c_xy梯度，大小为N * num_c (N * num_ann)
  const double* __restrict__ g_V_yz,  // V_c_yz梯度，大小为N * num_c (N * num_ann)
  const double* __restrict__ g_V_zx,  // V_c_zx梯度，大小为N * num_c (N * num_ann)
  double* __restrict__ g_gradients_sum,      // 每个结构的梯度之和，大小为num_c (num_ann)
  bool is_descriptor
)
{
  int sub_block_idx = blockIdx.y;
  if (sub_block_idx >= num_sub_blocks) return;

  int start_var = sub_block_idx * var_per_block;
  int end_var = min(start_var + var_per_block, num_var);
  
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (bid >= Nc) return;

  int Na = g_Na[bid];       
  int N1 = g_Na_sum[bid];   
  int N2 = N1 + Na;        
  
  const double per_Nc = 2.0 * lambda_v / count_Nc;
  double diff[6] = {
    g_diff[bid * 6 + 0],
    g_diff[bid * 6 + 1],
    g_diff[bid * 6 + 2],
    g_diff[bid * 6 + 3],
    g_diff[bid * 6 + 4],
    g_diff[bid * 6 + 5]
  };
  double weight = g_weight[bid];

  for (int c_idx = start_var + tid; c_idx < end_var; c_idx += blockDim.x) {
    if (c_idx >= num_var) continue;
    
    double V_xx = 0.0;
    double V_yy = 0.0;
    double V_zz = 0.0;
    double V_xy = 0.0;
    double V_yz = 0.0;
    double V_zx = 0.0;
    
    for (int n = N1; n < N2; ++n) {
      int index = is_descriptor ? c_idx * N + n : num_var * n + c_idx;
      V_xx -= g_V_xx[index];
      V_yy -= g_V_yy[index];
      V_zz -= g_V_zz[index];
      V_xy -= g_V_xy[index];
      V_yz -= g_V_yz[index];
      V_zx -= g_V_zx[index];
    }

    double temp = (V_xx * diff[0] + V_yy * diff[1] + V_zz * diff[2] +
                  V_xy * diff[3] + V_yz * diff[4] + V_zx * diff[5]) * weight * per_Nc;
    
    atomicAdd(&g_gradients_sum[c_idx], temp);
  }
}

std::vector<double> Dataset::get_rmse_virial(Parameters& para, const bool use_weight, const bool require_grad, int device_id)
{
  CHECK(cudaSetDevice(device_id));

  std::vector<double> rmse_array(para.num_types + 1, 0.0);
  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(double) * Nc;
  const int block_size = 256;

  double shear_weight =
    (para.train_mode != 1) ? (use_weight ? para.lambda_shear * para.lambda_shear : 1.0) : 0.0;
  gpu_sum_virial_error<<<Nc, block_size, sizeof(double) * block_size * 6>>>(
    N,
    shear_weight,
    Na.data(),
    Na_sum.data(),
    virial.data(),
    virial_ref_gpu.data(),
    diff_gpu_virial.data(),
    error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      double rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (has_type[t * Nc + n]) {
          rmse_array[t] += rmse_temp;
          count_array[t] += (para.train_mode != 1) ? 6 : 3;
        }
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      // rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
      rmse_array[t] = rmse_array[t] / count_array[t];
    }
  }
  if (count_array[para.num_types] > 0) {
    if (require_grad) {
      int num_sub_blocks = (para.number_of_variables_descriptor + block_size - 1) / block_size;
      dim3 grid_dim(Nc, num_sub_blocks);
      sum_gradients_v<<<grid_dim, block_size>>>(
        block_size, num_sub_blocks, N, Nc, 
        count_array[para.num_types], para.number_of_variables_descriptor, 
        para.lambda_v, Na.data(), Na_sum.data(), 
        diff_gpu_virial.data(), weight_gpu.data(), 
        gradients.V_c_xx.data(), gradients.V_c_yy.data(), 
        gradients.V_c_zz.data(), gradients.V_c_xy.data(), 
        gradients.V_c_yz.data(), gradients.V_c_zx.data(), 
        gradients.grad_c_sum.data(),
        true);
      CUDA_CHECK_KERNEL
  //     std::vector<double>grad_c_sum(para.number_of_variables_descriptor);
  //  CHECK(cudaMemcpy(grad_c_sum.data(), gradients.grad_c_sum.data(), para.number_of_variables_descriptor * sizeof(double), cudaMemcpyDeviceToHost));
  //  for (int j = 0; j < para.number_of_variables_descriptor; ++j) {
  //     printf("grad_c_sum[%d] = %f\n", j, grad_c_sum[j]);
  //   }

      int num_sub_blocks_wb = (para.number_of_variables_ann + block_size - 1) / block_size;
      dim3 grid_dim_wb(Nc, num_sub_blocks_wb);
      sum_gradients_v<<<grid_dim_wb, block_size>>>(
        block_size, num_sub_blocks_wb, N, Nc, 
        count_array[para.num_types], para.number_of_variables_ann, 
        para.lambda_v, Na.data(), Na_sum.data(), 
        diff_gpu_virial.data(), weight_gpu.data(), 
        gradients.V_wb_grad_xx.data(), gradients.V_wb_grad_yy.data(), 
        gradients.V_wb_grad_zz.data(), gradients.V_wb_grad_xy.data(), 
        gradients.V_wb_grad_yz.data(), gradients.V_wb_grad_zx.data(), 
        gradients.grad_wb_sum.data(),
        false);
      CUDA_CHECK_KERNEL
    // std::vector<double>grad_wb_sum(para.number_of_variables_ann);
    // CHECK(cudaMemcpy(grad_wb_sum.data(), gradients.grad_wb_sum.data(), para.number_of_variables_ann * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int j = 0; j < para.number_of_variables_ann; ++j) {
    //     printf("grad_wb_sum[%d] = %f\n", j, grad_wb_sum[j]);
    //   }
    }
  }
  return rmse_array;
}
