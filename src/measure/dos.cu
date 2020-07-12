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
  const int num_atoms, const int* g_group_contents, const double* g_mass_i, double* g_mass_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    g_mass_o[n] = g_mass_i[g_group_contents[n]];
  }
}

__global__ void gpu_copy_velocity(
  const int num_atoms,
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
    const int m = g_group_contents[n];
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

  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval_);

  if (!is_valid_int(param[2], &num_correlation_steps_)) {
    PRINT_INPUT_ERROR("number of correlation steps should be an integer.\n");
  }
  if (num_correlation_steps_ <= 0) {
    PRINT_INPUT_ERROR("number of correlation steps should be positive.\n");
  }
  printf("    number of correlation steps is %d.\n", num_correlation_steps_);

  if (!is_valid_real(param[3], &omega_max_)) {
    PRINT_INPUT_ERROR("maximal angular frequency should be a number.\n");
  }
  if (omega_max_ <= 0) {
    PRINT_INPUT_ERROR("maximal angular frequency should be positive.\n");
  }
  printf("    maximal angular frequency is %g THz.\n", omega_max_);

  for (int k = 4; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      if (k + 3 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for option 'group'.\n");
      }
      parse_group(param, groups, k, grouping_method_, group_id_);
      printf("    grouping_method is %d and group is %d.\n", grouping_method_, group_id_);
    } else if (strcmp(param[k], "num_dos_points") == 0) {
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

void DOS::preprocess(
  const double time_step, const std::vector<Group>& groups, const GPU_Vector<double>& mass)
{
  if (!compute_)
    return;
  initialize_parameters(time_step, groups, mass);
  allocate_memory();
  copy_mass(mass);
}

void DOS::process(
  const int step, const std::vector<Group>& groups, const GPU_Vector<double>& velocity_per_atom)
{
  if (!compute_)
    return;
  if ((step + 1) % sample_interval_ != 0)
    return;

  const int sample_step = step / sample_interval_;
  const int correlation_step = sample_step % num_correlation_steps_;
  copy_velocity(correlation_step, velocity_per_atom);
  if (sample_step >= num_correlation_steps_ - 1) {
    ++num_time_origins_;
    find_vac(correlation_step);
  }
}

void DOS::postprocess(const char* input_dir)
{
  if (!compute_)
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU

  normalize_vac();
  output_vac(input_dir);
  find_dos();
  output_dos(input_dir);

  compute_ = false;
  grouping_method_ = -1;
  num_dos_points_ = -1;
}

void DOS::parse_num_dos_points(char** param, int& k)
{
  if (!is_valid_int(param[k + 1], &num_dos_points_)) {
    PRINT_INPUT_ERROR("number of DOS points should be an integer.\n");
  }
  if (num_dos_points_ < 1) {
    PRINT_INPUT_ERROR("number of DOS points must be > 0.\n");
  }
  k += 1;
}

void DOS::initialize_parameters(
  const double time_step, const std::vector<Group>& groups, const GPU_Vector<double>& mass)
{
  num_time_origins_ = 0;
  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;
  if (1.0 / dt_in_ps_ < omega_max_ / PI) {
    PRINT_INPUT_ERROR("Velocity sampling rate < Nyquist frequency.");
  }

  if (grouping_method_ < 0) {
    num_atoms_ = mass.size();
    num_groups_ = 1;
  } else {
    group_ = &groups[grouping_method_];
    if (group_id_ < 0) {
      num_atoms_ = mass.size();
      num_groups_ = group_->number;
    } else {
      num_atoms_ = group_->cpu_size[group_id_];
      num_groups_ = 1;
    }
  }

  if (num_dos_points_ < 0) {
    num_dos_points_ = num_correlation_steps_;
  }
}

void DOS::allocate_memory()
{
  vx_.resize(num_atoms_ * num_correlation_steps_);
  vy_.resize(num_atoms_ * num_correlation_steps_);
  vz_.resize(num_atoms_ * num_correlation_steps_);
  vacx_.resize(num_groups_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  vacy_.resize(num_groups_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  vacz_.resize(num_groups_ * num_correlation_steps_, 0.0, Memory_Type::managed);
  dosx_.resize(num_groups_ * num_dos_points_, 0.0);
  dosy_.resize(num_groups_ * num_dos_points_, 0.0);
  dosz_.resize(num_groups_ * num_dos_points_, 0.0);
  mass_.resize(num_atoms_);
}

void DOS::copy_mass(const GPU_Vector<double>& mass)
{
  if (grouping_method_ < 0) {
    mass_.copy_from_device(mass.data());
  } else {
    const int offset = (group_id_ < 0) ? 0 : group_->cpu_size_sum[group_id_];
    gpu_copy_mass<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_, group_->contents.data() + offset, mass.data(), mass_.data());
    CUDA_CHECK_KERNEL
  }
}

void DOS::copy_velocity(const int correlation_step, const GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms_total = velocity_per_atom.size() / 3;
  const int step_offset = correlation_step * num_atoms_;
  const double* vxi = velocity_per_atom.data();
  const double* vyi = velocity_per_atom.data() + number_of_atoms_total;
  const double* vzi = velocity_per_atom.data() + number_of_atoms_total * 2;
  double* vxo = vx_.data() + step_offset;
  double* vyo = vy_.data() + step_offset;
  double* vzo = vz_.data() + step_offset;

  if (grouping_method_ < 0) {
    gpu_copy_velocity<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_, vxi, vyi, vzi, vxo, vyo, vzo);
  } else {
    if (group_id_ >= 0) {
      const int* group_contents = group_->contents.data() + group_->cpu_size_sum[group_id_];
      gpu_copy_velocity<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
        num_atoms_, group_contents, vxi, vyi, vzi, vxo, vyo, vzo);
    } else {
      for (int n = 0; n < num_groups_; ++n) {
        const int group_size = group_->cpu_size[n];
        const int step_offset =
          num_correlation_steps_ * group_->cpu_size_sum[n] + correlation_step * group_size;
        const int* group_contents = group_->contents.data() + group_->cpu_size_sum[n];
        vxo = vx_.data() + step_offset;
        vyo = vy_.data() + step_offset;
        vzo = vz_.data() + step_offset;
        gpu_copy_velocity<<<(group_size - 1) / 128 + 1, 128>>>(
          group_size, group_contents, vxi, vyi, vzi, vxo, vyo, vzo);
      }
    }
  }
  CUDA_CHECK_KERNEL
}

void DOS::find_vac(const int correlation_step)
{
  if (grouping_method_ >= 0 && group_id_ < 0) {
    for (int n = 0; n < num_groups_; ++n) {
      const int size_n = group_->cpu_size[n];
      const double* mass = mass_.data() + group_->cpu_size_sum[n];
      const double* vx_all = vx_.data() + num_correlation_steps_ * group_->cpu_size_sum[n];
      const double* vy_all = vy_.data() + num_correlation_steps_ * group_->cpu_size_sum[n];
      const double* vz_all = vz_.data() + num_correlation_steps_ * group_->cpu_size_sum[n];
      const double* vx = vx_all + correlation_step * size_n;
      const double* vy = vy_all + correlation_step * size_n;
      const double* vz = vz_all + correlation_step * size_n;
      double* vacx = vacx_.data() + num_correlation_steps_ * n;
      double* vacy = vacy_.data() + num_correlation_steps_ * n;
      double* vacz = vacz_.data() + num_correlation_steps_ * n;
      gpu_find_vac<<<num_correlation_steps_, 128>>>(
        size_n, correlation_step, mass, vx, vy, vz, vx_all, vy_all, vz_all, vacx, vacy, vacz);
    }
  } else {
    const int step_offset = correlation_step * num_atoms_;
    const double* vx = vx_.data() + step_offset;
    const double* vy = vy_.data() + step_offset;
    const double* vz = vz_.data() + step_offset;
    gpu_find_vac<<<num_correlation_steps_, 128>>>(
      num_atoms_, correlation_step, mass_.data(), vx, vy, vz, vx_.data(), vy_.data(), vz_.data(),
      vacx_.data(), vacy_.data(), vacz_.data());
  }
  CUDA_CHECK_KERNEL
}

void DOS::normalize_vac()
{
  for (int n = 0; n < num_groups_; ++n) {
    const int vac_offset = num_correlation_steps_ * n;
    const double vacx_0 = vacx_[vac_offset];
    const double vacy_0 = vacy_[vac_offset];
    const double vacz_0 = vacz_[vac_offset];
    for (int nc = 0; nc < num_correlation_steps_; nc++) {
      vacx_[nc + vac_offset] /= vacx_0;
      vacy_[nc + vac_offset] /= vacy_0;
      vacz_[nc + vac_offset] /= vacz_0;
    }
  }
}

void DOS::output_vac(const char* input_dir)
{
  char file_vac[200];
  strcpy(file_vac, input_dir);
  strcat(file_vac, "/mvac.out");
  FILE* fid = fopen(file_vac, "a");

  for (int n = 0; n < num_groups_; ++n) {
    const int offset = num_correlation_steps_ * n;
    for (int nc = 0; nc < num_correlation_steps_; nc++) {
      fprintf(
        fid, "%g %g %g %g\n", nc * dt_in_ps_, vacx_[nc + offset], vacy_[nc + offset],
        vacz_[nc + offset]);
    }
  }

  fflush(fid);
  fclose(fid);
}

void DOS::find_dos()
{
  const double d_omega = omega_max_ / num_dos_points_;

  for (int n = 0; n < num_groups_; ++n) {

    for (int nc = 0; nc < num_correlation_steps_; nc++) {
      const double hann_window = (cos((PI * nc) / num_correlation_steps_) + 1.0) * 0.5;
      const double multiply_factor = (nc == 0) ? hann_window : 2.0 * hann_window;
      vacx_[nc + num_correlation_steps_ * n] *= multiply_factor;
      vacy_[nc + num_correlation_steps_ * n] *= multiply_factor;
      vacz_[nc + num_correlation_steps_ * n] *= multiply_factor;
    }

    for (int nw = 0; nw < num_dos_points_; nw++) {
      const double omega = d_omega + nw * d_omega;

      for (int nc = 0; nc < num_correlation_steps_; nc++) {
        const double cos_factor = cos(omega * nc * dt_in_ps_);
        dosx_[nw + num_dos_points_ * n] += vacx_[nc + num_correlation_steps_ * n] * cos_factor;
        dosy_[nw + num_dos_points_ * n] += vacy_[nc + num_correlation_steps_ * n] * cos_factor;
        dosz_[nw + num_dos_points_ * n] += vacz_[nc + num_correlation_steps_ * n] * cos_factor;
      }

      const int group_size = (num_groups_ == 1) ? num_atoms_ : group_->cpu_size[n];

      dosx_[nw + num_dos_points_ * n] *= dt_in_ps_ * 2.0 * group_size;
      dosy_[nw + num_dos_points_ * n] *= dt_in_ps_ * 2.0 * group_size;
      dosz_[nw + num_dos_points_ * n] *= dt_in_ps_ * 2.0 * group_size;
    }
  }
}

void DOS::output_dos(const char* input_dir)
{
  char file_dos[200];
  strcpy(file_dos, input_dir);
  strcat(file_dos, "/dos.out");
  FILE* fid_dos = fopen(file_dos, "a");

  const double d_omega = omega_max_ / num_dos_points_;
  for (int ng = 0; ng < num_groups_; ++ng) {
    const int offset = num_dos_points_ * ng;
    for (int nw = 0; nw < num_dos_points_; nw++) {
      fprintf(
        fid_dos, "%g %g %g %g\n", d_omega + d_omega * nw, dosx_[nw + offset], dosy_[nw + offset],
        dosz_[nw + offset]);
    }
  }

  fflush(fid_dos);
  fclose(fid_dos);
}
