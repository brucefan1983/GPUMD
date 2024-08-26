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

/*-----------------------------------------------------------------------------------------------100
Calculate:
    Velocity AutoCorrelation (VAC) function
    Self Diffusion Coefficient (SDC)
--------------------------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "sdc.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

namespace
{

__global__ void gpu_copy_velocity(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    const int m = g_group_contents[offset + n];
    g_vx_o[n] = g_vx_i[m];
    g_vy_o[n] = g_vy_i[m];
    g_vz_o[n] = g_vz_i[m];
  }
}

__global__ void gpu_copy_velocity(
  const int num_atoms,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    g_vx_o[n] = g_vx_i[n];
    g_vy_o[n] = g_vy_i[n];
    g_vz_o[n] = g_vz_i[n];
  }
}

__global__ void gpu_find_vac(
  const int num_atoms,
  const int correlation_step,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_vx_all,
  const double* g_vy_all,
  const double* g_vz_all,
  double* g_vac_x,
  double* g_vac_y,
  double* g_vac_z)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_sum = bid * num_atoms;
  int number_of_rounds = (num_atoms - 1) / 128 + 1;
  __shared__ double s_vac_x[128];
  __shared__ double s_vac_y[128];
  __shared__ double s_vac_z[128];
  double vac_x = 0.0;
  double vac_y = 0.0;
  double vac_z = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * 128;
    if (n < num_atoms) {
      vac_x += g_vx[n] * g_vx_all[size_sum + n];
      vac_y += g_vy[n] * g_vy_all[size_sum + n];
      vac_z += g_vz[n] * g_vz_all[size_sum + n];
    }
  }
  s_vac_x[tid] = vac_x;
  s_vac_y[tid] = vac_y;
  s_vac_z[tid] = vac_z;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_vac_x[tid] += s_vac_x[tid + offset];
      s_vac_y[tid] += s_vac_y[tid + offset];
      s_vac_z[tid] += s_vac_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (bid <= correlation_step) {
      g_vac_x[correlation_step - bid] += s_vac_x[0];
      g_vac_y[correlation_step - bid] += s_vac_y[0];
      g_vac_z[correlation_step - bid] += s_vac_z[0];
    } else {
      g_vac_x[correlation_step + gridDim.x - bid] += s_vac_x[0];
      g_vac_y[correlation_step + gridDim.x - bid] += s_vac_y[0];
      g_vac_z[correlation_step + gridDim.x - bid] += s_vac_z[0];
    }
  }
}

} // namespace

void SDC::preprocess(const int num_atoms, const double time_step, const std::vector<Group>& groups)
{
  if (!compute_)
    return;

  num_atoms_ = (grouping_method_ < 0) ? num_atoms : groups[grouping_method_].cpu_size[group_id_];
  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;
  vx_.resize(num_atoms_ * num_correlation_steps_);
  vy_.resize(num_atoms_ * num_correlation_steps_);
  vz_.resize(num_atoms_ * num_correlation_steps_);
  vacx_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);
  vacy_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);
  vacz_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);

  num_time_origins_ = 0;
}

void SDC::process(
  const int step, const std::vector<Group>& groups, const GPU_Vector<double>& velocity_per_atom)
{
  if (!compute_)
    return;
  if ((step + 1) % sample_interval_ != 0)
    return;

  const int sample_step = step / sample_interval_;
  const int correlation_step = sample_step % num_correlation_steps_;
  const int step_offset = correlation_step * num_atoms_;
  const int number_of_atoms_total = velocity_per_atom.size() / 3;

  // copy the velocity data at the current step to appropriate place
  if (grouping_method_ < 0) {
    gpu_copy_velocity<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_,
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total,
      vx_.data() + step_offset,
      vy_.data() + step_offset,
      vz_.data() + step_offset);
  } else {
    const int group_offset = groups[grouping_method_].cpu_size_sum[group_id_];
    gpu_copy_velocity<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_,
      group_offset,
      groups[grouping_method_].contents.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total,
      vx_.data() + step_offset,
      vy_.data() + step_offset,
      vz_.data() + step_offset);
  }
  CUDA_CHECK_KERNEL

  // start to calculate the VAC when we have enough frames
  if (sample_step >= num_correlation_steps_ - 1) {
    ++num_time_origins_;

    gpu_find_vac<<<num_correlation_steps_, 128>>>(
      num_atoms_,
      correlation_step,
      vx_.data() + step_offset,
      vy_.data() + step_offset,
      vz_.data() + step_offset,
      vx_.data(),
      vy_.data(),
      vz_.data(),
      vacx_.data(),
      vacy_.data(),
      vacz_.data());
    CUDA_CHECK_KERNEL
  }
}

void SDC::postprocess()
{
  if (!compute_)
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU

  // normalize by the number of atoms and number of time origins
  const double vac_scaler = 1.0 / ((double)num_atoms_ * (double)num_time_origins_);
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    vacx_[nc] *= vac_scaler;
    vacy_[nc] *= vac_scaler;
    vacz_[nc] *= vac_scaler;
  }

  std::vector<double> sdc_x(num_correlation_steps_, 0.0);
  std::vector<double> sdc_y(num_correlation_steps_, 0.0);
  std::vector<double> sdc_z(num_correlation_steps_, 0.0);
  const double dt2 = dt_in_natural_units_ * 0.5;
  for (int nc = 1; nc < num_correlation_steps_; nc++) {
    sdc_x[nc] = sdc_x[nc - 1] + (vacx_[nc - 1] + vacx_[nc]) * dt2;
    sdc_y[nc] = sdc_y[nc - 1] + (vacy_[nc - 1] + vacy_[nc]) * dt2;
    sdc_z[nc] = sdc_z[nc - 1] + (vacz_[nc - 1] + vacz_[nc]) * dt2;
  }

  const double sdc_unit_conversion = 1.0e3 / TIME_UNIT_CONVERSION;
  const double vac_unit_conversion = sdc_unit_conversion * sdc_unit_conversion;

  FILE* fid = fopen("sdc.out", "a");
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    fprintf(
      fid,
      "%g %g %g %g %g %g %g\n",
      nc * dt_in_ps_,
      vacx_[nc] * vac_unit_conversion,
      vacy_[nc] * vac_unit_conversion,
      vacz_[nc] * vac_unit_conversion,
      sdc_x[nc] * sdc_unit_conversion,
      sdc_y[nc] * sdc_unit_conversion,
      sdc_z[nc] * sdc_unit_conversion);
  }
  fflush(fid);
  fclose(fid);

  compute_ = false;
  grouping_method_ = -1;
}

void SDC::parse(const char** param, const int num_param, const std::vector<Group>& groups)
{
  printf("Compute self diffusion coefficient (SDC).\n");
  compute_ = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("compute_sdc should have at least 2 parameters.\n");
  }
  if (num_param > 6) {
    PRINT_INPUT_ERROR("compute_sdc has too many parameters.\n");
  }

  // sample interval
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval_);

  // number of correlation steps
  if (!is_valid_int(param[2], &num_correlation_steps_)) {
    PRINT_INPUT_ERROR("number of correlation steps should be an integer.\n");
  }
  if (num_correlation_steps_ <= 0) {
    PRINT_INPUT_ERROR("number of correlation steps should be positive.\n");
  }
  printf("    number of correlation steps is %d.\n", num_correlation_steps_);

  // Process optional arguments
  for (int k = 3; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, num_param, false, groups, k, grouping_method_, group_id_);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_sdc.\n");
    }
  }
}
