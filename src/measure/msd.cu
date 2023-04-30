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
    Mean-Square Displacement (MSD)
    Self Diffusion Coefficient (SDC)
--------------------------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "msd.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"

namespace
{

__global__ void gpu_copy_position(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  const double* g_x_i,
  const double* g_y_i,
  const double* g_z_i,
  double* g_x_o,
  double* g_y_o,
  double* g_z_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    const int m = g_group_contents[offset + n];
    g_x_o[n] = g_x_i[m];
    g_y_o[n] = g_y_i[m];
    g_z_o[n] = g_z_i[m];
  }
}

__global__ void gpu_copy_position(
  const int num_atoms,
  const double* g_x_i,
  const double* g_y_i,
  const double* g_z_i,
  double* g_x_o,
  double* g_y_o,
  double* g_z_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    g_x_o[n] = g_x_i[n];
    g_y_o[n] = g_y_i[n];
    g_z_o[n] = g_z_i[n];
  }
}

__global__ void gpu_find_msd(
  const int num_atoms,
  const int correlation_step,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const double* g_x_all,
  const double* g_y_all,
  const double* g_z_all,
  double* g_msd_x,
  double* g_msd_y,
  double* g_msd_z)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_sum = bid * num_atoms;
  int number_of_rounds = (num_atoms - 1) / 128 + 1;
  __shared__ double s_msd_x[128];
  __shared__ double s_msd_y[128];
  __shared__ double s_msd_z[128];
  double msd_x = 0.0;
  double msd_y = 0.0;
  double msd_z = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * 128;
    if (n < num_atoms) {
      double tmp = g_x[n] - g_x_all[size_sum + n];
      msd_x += tmp * tmp;
      tmp = g_y[n] - g_y_all[size_sum + n];
      msd_y += tmp * tmp;
      tmp = g_z[n] - g_z_all[size_sum + n];
      msd_z += tmp * tmp;
    }
  }
  s_msd_x[tid] = msd_x;
  s_msd_y[tid] = msd_y;
  s_msd_z[tid] = msd_z;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_msd_x[tid] += s_msd_x[tid + offset];
      s_msd_y[tid] += s_msd_y[tid + offset];
      s_msd_z[tid] += s_msd_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (bid <= correlation_step) {
      g_msd_x[correlation_step - bid] += s_msd_x[0];
      g_msd_y[correlation_step - bid] += s_msd_y[0];
      g_msd_z[correlation_step - bid] += s_msd_z[0];
    } else {
      g_msd_x[correlation_step + gridDim.x - bid] += s_msd_x[0];
      g_msd_y[correlation_step + gridDim.x - bid] += s_msd_y[0];
      g_msd_z[correlation_step + gridDim.x - bid] += s_msd_z[0];
    }
  }
}

} // namespace

void MSD::preprocess(const int num_atoms, const double time_step, const std::vector<Group>& groups)
{
  if (!compute_)
    return;

  num_atoms_ = (grouping_method_ < 0) ? num_atoms : groups[grouping_method_].cpu_size[group_id_];
  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;
  x_.resize(num_atoms_ * num_correlation_steps_);
  y_.resize(num_atoms_ * num_correlation_steps_);
  z_.resize(num_atoms_ * num_correlation_steps_);
  msdx_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);
  msdy_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);
  msdz_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);

  num_time_origins_ = 0;
}

void MSD::process(const int step, const std::vector<Group>& groups, const GPU_Vector<double>& xyz)
{
  if (!compute_)
    return;
  if ((step + 1) % sample_interval_ != 0)
    return;

  const int sample_step = step / sample_interval_;
  const int correlation_step = sample_step % num_correlation_steps_;
  const int step_offset = correlation_step * num_atoms_;
  const int number_of_atoms_total = xyz.size() / 3;

  // copy the position data at the current step to appropriate place
  if (grouping_method_ < 0) {
    gpu_copy_position<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_,
      xyz.data(),
      xyz.data() + number_of_atoms_total,
      xyz.data() + 2 * number_of_atoms_total,
      x_.data() + step_offset,
      y_.data() + step_offset,
      z_.data() + step_offset);
  } else {
    const int group_offset = groups[grouping_method_].cpu_size_sum[group_id_];
    gpu_copy_position<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_,
      group_offset,
      groups[grouping_method_].contents.data(),
      xyz.data(),
      xyz.data() + number_of_atoms_total,
      xyz.data() + 2 * number_of_atoms_total,
      x_.data() + step_offset,
      y_.data() + step_offset,
      z_.data() + step_offset);
  }
  CUDA_CHECK_KERNEL

  // start to calculate the MSD when we have enough frames
  if (sample_step >= num_correlation_steps_ - 1) {
    ++num_time_origins_;

    gpu_find_msd<<<num_correlation_steps_, 128>>>(
      num_atoms_,
      correlation_step,
      x_.data() + step_offset,
      y_.data() + step_offset,
      z_.data() + step_offset,
      x_.data(),
      y_.data(),
      z_.data(),
      msdx_.data(),
      msdy_.data(),
      msdz_.data());
    CUDA_CHECK_KERNEL
  }
}

void MSD::postprocess()
{
  if (!compute_)
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU

  // normalize by the number of atoms and number of time origins
  const double msd_scaler = 1.0 / ((double)num_atoms_ * (double)num_time_origins_);
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    msdx_[nc] *= msd_scaler;
    msdy_[nc] *= msd_scaler;
    msdz_[nc] *= msd_scaler;
  }

  std::vector<double> sdc_x(num_correlation_steps_, 0.0);
  std::vector<double> sdc_y(num_correlation_steps_, 0.0);
  std::vector<double> sdc_z(num_correlation_steps_, 0.0);
  const double dt2inv = 0.5 / dt_in_natural_units_;
  for (int nc = 1; nc < num_correlation_steps_; nc++) {
    sdc_x[nc] = (msdx_[nc] - msdx_[nc - 1]) * dt2inv;
    sdc_y[nc] = (msdy_[nc] - msdy_[nc - 1]) * dt2inv;
    sdc_z[nc] = (msdz_[nc] - msdz_[nc - 1]) * dt2inv;
  }

  const double sdc_unit_conversion = 1.0e3 / TIME_UNIT_CONVERSION;

  FILE* fid = fopen("msd.out", "a");
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    fprintf(
      fid,
      "%g %g %g %g %g %g %g\n",
      nc * dt_in_ps_,
      msdx_[nc],
      msdy_[nc],
      msdz_[nc],
      sdc_x[nc] * sdc_unit_conversion,
      sdc_y[nc] * sdc_unit_conversion,
      sdc_z[nc] * sdc_unit_conversion);
  }
  fflush(fid);
  fclose(fid);

  compute_ = false;
  grouping_method_ = -1;
}

void MSD::parse(const char** param, const int num_param, const std::vector<Group>& groups)
{
  printf("Compute mean square displacement (MSD).\n");
  compute_ = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("compute_msd should have at least 2 parameters.\n");
  }
  if (num_param > 6) {
    PRINT_INPUT_ERROR("compute_msd has too many parameters.\n");
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
      PRINT_INPUT_ERROR("Unrecognized argument in compute_msd.\n");
    }
  }
}
