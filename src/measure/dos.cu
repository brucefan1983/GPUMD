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
    Mass-weighted Velocity AutoCorrelation (MVAC)
    Density Of States (DOS)

Reference for DOS:
    J. M. Dickey and A. Paskin,
    Computer Simulation of the Lattice Dynamics of Solids,
    Phys. Rev. 188, 1407 (1969).
--------------------------------------------------------------------------------------------------*/

#include "dos.cuh"
#include "model/group.cuh"
#include "parse_group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"

namespace
{
__global__ void gpu_copy_mass(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  const double* g_mass_i,
  double* g_mass_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    g_mass_o[n] = g_mass_i[g_group_contents[offset + n]];
  }
}

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
  const double* g_mass,
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
      double mass = g_mass[n];
      vac_x += mass * g_vx[n] * g_vx_all[size_sum + n];
      vac_y += mass * g_vy[n] * g_vy_all[size_sum + n];
      vac_z += mass * g_vz[n] * g_vz_all[size_sum + n];
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

void DOS::preprocess(
  const double time_step, const std::vector<Group>& groups, const GPU_Vector<double>& mass)
{
  if (!compute_)
    return;

  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;

  if (1.0 / dt_in_ps_ < omega_max_ / PI) {
    PRINT_INPUT_ERROR("Velocity sampling rate < Nyquist frequency.");
  }

  if (grouping_method_ < 0) {
    num_atoms_ = mass.size();
    num_groups_ = 1;
  } else {
    if (group_id_ < 0) {
      num_atoms_ = mass.size();
      num_groups_ = groups[grouping_method_].number;
    } else {
      num_atoms_ = groups[grouping_method_].cpu_size[group_id_];
      num_groups_ = 1;
    }
  }

  if (num_dos_points_ < 0) {
    num_dos_points_ = num_correlation_steps_;
  }

  vx_.resize(num_atoms_ * num_correlation_steps_);
  vy_.resize(num_atoms_ * num_correlation_steps_);
  vz_.resize(num_atoms_ * num_correlation_steps_);
  vacx_.resize(num_groups_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  vacy_.resize(num_groups_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  vacz_.resize(num_groups_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  mass_.resize(num_atoms_);

  if (grouping_method_ < 0) {
    mass_.copy_from_device(mass.data());
  } else {
    const int offset = (group_id_ < 0) ? 0 : groups[grouping_method_].cpu_size_sum[group_id_];
    gpu_copy_mass<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_, offset, groups[grouping_method_].contents.data(), mass.data(), mass_.data());
    CUDA_CHECK_KERNEL
  }

  num_time_origins_ = 0;
}

void DOS::process(
  const int step, const std::vector<Group>& groups, const GPU_Vector<double>& velocity_per_atom)
{
  if (!compute_)
    return;
  if ((step + 1) % sample_interval_ != 0) {
    return;
  }
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
    const int group_offset = (group_id_ < 0) ? 0 : groups[grouping_method_].cpu_size_sum[group_id_];
    gpu_copy_velocity<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_, group_offset, groups[grouping_method_].contents.data(), velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms_total,
      velocity_per_atom.data() + 2 * number_of_atoms_total, vx_.data() + step_offset,
      vy_.data() + step_offset, vz_.data() + step_offset);
  }
  CUDA_CHECK_KERNEL

  // start to calculate the MVAC when we have enough frames
  if (sample_step >= num_correlation_steps_ - 1) {
    ++num_time_origins_;

    gpu_find_vac<<<num_correlation_steps_, 128>>>(
      num_atoms_, correlation_step, mass_.data(), vx_.data() + step_offset,
      vy_.data() + step_offset, vz_.data() + step_offset, vx_.data(), vy_.data(), vz_.data(),
      vacx_.data(), vacy_.data(), vacz_.data());
    CUDA_CHECK_KERNEL
  }
}

void DOS::postprocess(const char* input_dir)
{
  if (!compute_)
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU

  // normalize MVAC
  double vacx_0 = vacx_[0];
  double vacy_0 = vacy_[0];
  double vacz_0 = vacz_[0];
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    vacx_[nc] /= vacx_0;
    vacy_[nc] /= vacy_0;
    vacz_[nc] /= vacz_0;
  }

  // output normalized MVAC
  char file_vac[200];
  strcpy(file_vac, input_dir);
  strcat(file_vac, "/mvac.out");
  FILE* fid = fopen(file_vac, "a");
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    double t = nc * dt_in_ps_;
    fprintf(fid, "%g %g %g %g\n", t, vacx_[nc], vacy_[nc], vacz_[nc]);
  }
  fflush(fid);
  fclose(fid);

  // calculate DOS
  double d_omega = omega_max_ / num_dos_points_;
  double omega_0 = d_omega;
  std::vector<double> dos_x(num_dos_points_, 0.0);
  std::vector<double> dos_y(num_dos_points_, 0.0);
  std::vector<double> dos_z(num_dos_points_, 0.0);
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    double hann_window = (cos((PI * nc) / num_correlation_steps_) + 1.0) * 0.5;
    double multiply_factor = (nc == 0) ? 1.0 * hann_window : 2.0 * hann_window;
    vacx_[nc] *= multiply_factor;
    vacy_[nc] *= multiply_factor;
    vacz_[nc] *= multiply_factor;
  }
  for (int nw = 0; nw < num_dos_points_; nw++) {
    double omega = omega_0 + nw * d_omega;
    for (int nc = 0; nc < num_correlation_steps_; nc++) {
      double cos_factor = cos(omega * nc * dt_in_ps_);
      dos_x[nw] += vacx_[nc] * cos_factor;
      dos_y[nw] += vacy_[nc] * cos_factor;
      dos_z[nw] += vacz_[nc] * cos_factor;
    }
    dos_x[nw] *= dt_in_ps_ * 2.0 * num_atoms_;
    dos_y[nw] *= dt_in_ps_ * 2.0 * num_atoms_;
    dos_z[nw] *= dt_in_ps_ * 2.0 * num_atoms_;
  }

  // output DOS
  char file_dos[200];
  strcpy(file_dos, input_dir);
  strcat(file_dos, "/dos.out");
  FILE* fid_dos = fopen(file_dos, "a");
  for (int nw = 0; nw < num_dos_points_; nw++) {
    double omega = omega_0 + d_omega * nw;
    fprintf(fid_dos, "%g %g %g %g\n", omega, dos_x[nw], dos_y[nw], dos_z[nw]);
  }
  fflush(fid_dos);
  fclose(fid_dos);

  compute_ = false;
  grouping_method_ = -1;
  group_id_ = -1;
  num_dos_points_ = -1;
}

void DOS::parse_num_dos_points(char** param, int& k)
{
  // number of DOS points
  if (!is_valid_int(param[k + 1], &num_dos_points_)) {
    PRINT_INPUT_ERROR("number of DOS points for VAC should be an integer.\n");
  }
  if (num_dos_points_ < 1) {
    PRINT_INPUT_ERROR("number of DOS points for DOS must be > 0.\n");
  }
  k += 1;
}

void DOS::parse(char** param, const int num_param, const std::vector<Group>& groups)
{
  printf("Compute phonon DOS.\n");
  compute_ = true;

  if (num_param < 4) {
    PRINT_INPUT_ERROR("compute_dos should have at least 3 parameters.\n");
  }
  if (num_param > 9) {
    PRINT_INPUT_ERROR("compute_dos has too many parameters.\n");
  }

  // sample interval
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval for VAC should be an integer number.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval for VAC should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval_);

  // number of correlation steps
  if (!is_valid_int(param[2], &num_correlation_steps_)) {
    PRINT_INPUT_ERROR("Nc for VAC should be an integer number.\n");
  }
  if (num_correlation_steps_ <= 0) {
    PRINT_INPUT_ERROR("Nc for VAC should be positive.\n");
  }
  printf("    Nc is %d.\n", num_correlation_steps_);

  // maximal omega
  if (!is_valid_real(param[3], &omega_max_)) {
    PRINT_INPUT_ERROR("omega_max should be a real number.\n");
  }
  if (omega_max_ <= 0) {
    PRINT_INPUT_ERROR("omega_max should be positive.\n");
  }
  printf("    omega_max is %g THz.\n", omega_max_);

  // Process optional arguments
  for (int k = 4; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      // check if there are enough inputs
      if (k + 3 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for option 'group'.\n");
      }
      parse_group(param, groups, k, grouping_method_, group_id_);
      printf("    grouping_method is %d and group is %d.\n", grouping_method_, group_id_);
    } else if (strcmp(param[k], "num_dos_points") == 0) {
      // check if there are enough inputs
      if (k + 2 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for option 'num_dos_points'.\n");
      }
      parse_num_dos_points(param, k);
      printf("    num_dos_points is %d.\n", num_dos_points_);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_dos.\n");
    }
  }
}
