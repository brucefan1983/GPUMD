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
    Ionic conductivity using the Einstein relation, based on the mean squared displacement.
--------------------------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "iron_conductivity.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

namespace
{
__global__ void gpu_copy_position(
  const int num_atoms,
  const int* g_atom_list,
  const double* g_x_i,
  const double* g_y_i,
  const double* g_z_i,
  double* g_x_o,
  double* g_y_o,
  double* g_z_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    const int m = g_atom_list[n];
    g_x_o[n] = g_x_i[m];
    g_y_o[n] = g_y_i[m];
    g_z_o[n] = g_z_i[m];
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
} //namespace

void IC::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& groups, 
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!compute_)
    return;
  
  if (num_correlation_steps_ > number_of_steps / sample_interval_) {
    PRINT_INPUT_ERROR("Correlation times sampling interval should be <= number of MD steps.\n");
  }
    
  std::vector<int> temp_list;
  temp_list.reserve(atom.number_of_atoms);
  
  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_type[i] == target_type_) {
      temp_list.push_back(i);
    }
  }
  
  num_atoms_ = temp_list.size();
  if (num_atoms_ == 0) {
    PRINT_INPUT_ERROR("No atoms found for the specified type.\n");
  }
  
  type_atom_list_.resize(num_atoms_);
  type_atom_list_.copy_from_host(temp_list.data());
  
  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;
  x_.resize(num_atoms_ * num_correlation_steps_);
  y_.resize(num_atoms_ * num_correlation_steps_);
  z_.resize(num_atoms_ * num_correlation_steps_);
  msdx_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);
  msdy_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);
  msdz_.resize(num_correlation_steps_, 0.0, Memory_Type::managed);

  volume_ = box.get_volume();
  num_time_origins_ = 0;
}

void IC::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& groups, 
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if (!compute_)
    return;
  temperature_ = temperature;
  if ((step + 1) % sample_interval_ != 0)
    return;

  const int sample_step = step / sample_interval_;
  const int correlation_step = sample_step % num_correlation_steps_;
  const int step_offset = correlation_step * num_atoms_;
  const int number_of_atoms_total = atom.number_of_atoms;

  gpu_copy_position<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
    num_atoms_,
    type_atom_list_.data(), 
    atom.unwrapped_position.data(),
    atom.unwrapped_position.data() + number_of_atoms_total,
    atom.unwrapped_position.data() + 2 * number_of_atoms_total,
    x_.data() + step_offset,
    y_.data() + step_offset,
    z_.data() + step_offset);
  GPU_CHECK_KERNEL

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
    GPU_CHECK_KERNEL
  }
}

void IC::write(const char* filename)
{
  CHECK(gpuDeviceSynchronize());
  
  std::vector<double> ic_x(num_correlation_steps_, 0.0);
  std::vector<double> ic_y(num_correlation_steps_, 0.0);
  std::vector<double> ic_z(num_correlation_steps_, 0.0);

  double factor = 0.0;
  if (num_time_origins_ > 0) {
    factor = charge_ * charge_ * 1.602176634e7 * 0.5 / 
            (TIME_UNIT_CONVERSION * volume_ * K_B * temperature_ * num_time_origins_ * dt_in_natural_units_);
  }

  for (int nc = 1; nc < num_correlation_steps_; nc++) {
    ic_x[nc] = (msdx_[nc] - msdx_[nc - 1]) * factor;
    ic_y[nc] = (msdy_[nc] - msdy_[nc - 1]) * factor;
    ic_z[nc] = (msdz_[nc] - msdz_[nc - 1]) * factor;
  }

  FILE* fid = fopen(filename, "a");
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    fprintf(fid, "%g %g %g %g\n",
      nc * dt_in_ps_,
      ic_x[nc],
      ic_y[nc],
      ic_z[nc]);
  }
  fflush(fid);
  fclose(fid);
}

void IC::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute_)
    return;
  write("ic.out");
  compute_ = false;
}

IC::IC(const char** param, const int num_param, Atom& atom)
{
  parse(param, num_param);
  if (atom.unwrapped_position.size() < atom.number_of_atoms * 3) {
    atom.unwrapped_position.resize(atom.number_of_atoms * 3);
    atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
  }
  if (atom.position_temp.size() < atom.number_of_atoms * 3) {
    atom.position_temp.resize(atom.number_of_atoms * 3);
  }
  property_name = "compute_ic";
}

void IC::parse(const char** param, const int num_param)
{
  printf("Compute ionic conductivity.\n");
  compute_ = true;

  if (num_param != 5) {
    PRINT_INPUT_ERROR("compute_ic should have exactly 5 parameters.\n");
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

  if (!is_valid_int(param[3], &target_type_)) {
    PRINT_INPUT_ERROR("type should be an integer.\n");
  }
  if (target_type_ < 0) {
    PRINT_INPUT_ERROR("type should be non-negative.\n");
  }
  printf("    will compute conductivity for atom type = %d.\n", target_type_);

  if (!is_valid_real(param[4], &charge_)) {
    PRINT_INPUT_ERROR("charge should be a real number.\n");
  }
  printf("    will compute conductivity using charge = %g.\n", charge_);
}
