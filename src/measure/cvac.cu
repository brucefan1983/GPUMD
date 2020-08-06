/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
    Cross Velocity AutoCorrelation (CVAC) function
--------------------------------------------------------------------------------------------------*/

#include "cvac.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"

namespace
{

__global__ void gpu_copy_velocity(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i,
  float* g_vx_o,
  float* g_vy_o,
  float* g_vz_o)
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
  float* g_vx_o,
  float* g_vy_o,
  float* g_vz_o)
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
  const int num_correlation_steps,
  const float* g_vx,
  const float* g_vy,
  const float* g_vz,
  const float* g_vx_all,
  const float* g_vy_all,
  const float* g_vz_all,
  float* g_vac_x,
  float* g_vac_y,
  float* g_vac_z)
{
  const int num_atoms_sq = num_atoms * num_atoms;
  const int n1n2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1n2 >= num_atoms_sq)
    return;
  const int n1 = n1n2 / num_atoms;
  const int n2 = n1n2 - n1 * num_atoms;
  for (int k = 0; k < num_correlation_steps; ++k) {
    int nc = correlation_step - k;
    if (nc < 0)
      nc += num_correlation_steps;
    g_vac_x[nc * num_atoms_sq + n1n2] += g_vx[n1] * g_vx_all[k * num_atoms + n2];
    g_vac_y[nc * num_atoms_sq + n1n2] += g_vy[n1] * g_vy_all[k * num_atoms + n2];
    g_vac_z[nc * num_atoms_sq + n1n2] += g_vz[n1] * g_vz_all[k * num_atoms + n2];
  }
}

} // namespace

void CVAC::preprocess(const int num_atoms, const double time_step, const std::vector<Group>& groups)
{
  if (!compute_)
    return;

  num_atoms_ = (grouping_method_ < 0) ? num_atoms : groups[grouping_method_].cpu_size[group_id_];
  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;
  vx_.resize(num_atoms_ * num_correlation_steps_);
  vy_.resize(num_atoms_ * num_correlation_steps_);
  vz_.resize(num_atoms_ * num_correlation_steps_);
  vacx_.resize(num_atoms_ * num_atoms_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  vacy_.resize(num_atoms_ * num_atoms_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  vacz_.resize(num_atoms_ * num_atoms_ * num_correlation_steps_, 0.0, Memory_Type::managed);

  num_time_origins_ = 0;
}

void CVAC::process(
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
      num_atoms_, velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total, vx_.data() + step_offset,
      vy_.data() + step_offset, vz_.data() + step_offset);
  } else {
    const int group_offset = groups[grouping_method_].cpu_size_sum[group_id_];
    gpu_copy_velocity<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_, group_offset, groups[grouping_method_].contents.data(), velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total, vx_.data() + step_offset,
      vy_.data() + step_offset, vz_.data() + step_offset);
  }
  CUDA_CHECK_KERNEL

  // start to calculate the VAC when we have enough frames
  if (sample_step >= num_correlation_steps_ - 1) {
    ++num_time_origins_;

    gpu_find_vac<<<(num_atoms_ * num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_, correlation_step, num_correlation_steps_, vx_.data() + step_offset,
      vy_.data() + step_offset, vz_.data() + step_offset, vx_.data(), vy_.data(), vz_.data(),
      vacx_.data(), vacy_.data(), vacz_.data());
    CUDA_CHECK_KERNEL
  }
}

void CVAC::postprocess(const char* input_dir)
{
  if (!compute_)
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU

  float vac_unit_conversion = 1.0e3 / TIME_UNIT_CONVERSION;
  vac_unit_conversion *= vac_unit_conversion;

  char file_cvac[200];
  strcpy(file_cvac, input_dir);
  strcat(file_cvac, "/cvac.out");
  FILE* fid = fopen(file_cvac, "a");
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    for (int n = 0; n < num_atoms_ * num_atoms_; ++n) {
      const int i = nc * num_atoms_ * num_atoms_ + n;
      fprintf(fid, "%g %g %g\n", vacx_[i], vacy_[i], vacz_[i]);
    }
  }
  fflush(fid);
  fclose(fid);

  compute_ = false;
  grouping_method_ = -1;
}

void CVAC::parse(char** param, const int num_param, const std::vector<Group>& groups)
{
  printf("Compute cross velocity auto-correlation (CVAC).\n");
  compute_ = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("compute_cvac should have at least 2 parameters.\n");
  }
  if (num_param > 6) {
    PRINT_INPUT_ERROR("compute_cvac has too many parameters.\n");
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
      PRINT_INPUT_ERROR("Unrecognized argument in compute_cvac.\n");
    }
  }
}
