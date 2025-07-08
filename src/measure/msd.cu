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
    Mean-Square Displacement (MSD)
    Self Diffusion Coefficient (SDC)
--------------------------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "msd.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>


#ifdef USE_KEPLER
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif
#endif

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

  __global__ void gpu_find_msd_per_group(
    const int num_atoms,
    const int num_groups,
    const int correlation_step,
    const int num_correlation_steps,
    const int* group_ids,
    const double* g_x,
    const double* g_y,
    const double* g_z,
    const double* g_x_start,
    const double* g_y_start,
    const double* g_z_start,
    double* g_msd_x,  // [num_groups * time_lags]
    double* g_msd_y,
    double* g_msd_z)
  {
    // Do not use a reduction, instead handle each atom independently.
    // Reduction does not work well, since atoms in the same group are not necessarily
    // close in memory.

    // Rewrite as a reduction over each group?
    // Need the size_sum! Sum 0 to l, 1 to l, etc.
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int size_sum = bid * num_atoms;
    int number_of_rounds = (num_atoms - 1) / blockDim.x + 1;  // rounds needed to sum over all atoms

    for (int round = 0; round < number_of_rounds; ++round) {
      int n = tid + round * blockDim.x;
      if (n < num_atoms) {

        int group = group_ids[n];
        if (group < 0 || group >= num_groups) return;

        double dx = g_x[n] - g_x_start[size_sum + n];
        double dy = g_y[n] - g_y_start[size_sum + n];
        double dz = g_z[n] - g_z_start[size_sum + n];

        if (bid <= correlation_step) {
          int offset = group * num_correlation_steps + correlation_step - bid;
          atomicAdd(&g_msd_x[offset], dx * dx);
          atomicAdd(&g_msd_y[offset], dy * dy);
          atomicAdd(&g_msd_z[offset], dz * dz);
        } else {
          int offset = group * num_correlation_steps + correlation_step + gridDim.x - bid;
          atomicAdd(&g_msd_x[offset], dx * dx);
          atomicAdd(&g_msd_y[offset], dy * dy);
          atomicAdd(&g_msd_z[offset], dz * dz);
        }
      }
    }
  }
} //namespace

void MSD::preprocess(
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
  
  if (num_correlation_steps_ > number_of_steps) {
    PRINT_INPUT_ERROR("MSD correlation should be <= number of MD steps.\n");
  }
    
  if (grouping_method_ < 0) {
    num_atoms_ = atom.number_of_atoms;
    num_groups_ = 1;
    num_atoms_per_group_.resize(1);
    num_atoms_per_group_[0] = num_atoms_;
  } else if (msd_over_all_groups_) {
    num_atoms_ = atom.number_of_atoms;
    num_groups_ = groups[grouping_method_].number;
    num_atoms_per_group_.resize(num_groups_);
    group_per_atom_cpu_.resize(num_atoms_);
    group_per_atom_gpu_.resize(num_atoms_);
    
    for (int i=0; i<num_groups_; i++) {
      int atoms_in_group = groups[grouping_method_].cpu_size[i];
      int group_offset = groups[grouping_method_].cpu_size_sum[i];
      num_atoms_per_group_[i] = atoms_in_group;

      for (int j=0; j<atoms_in_group; j++) {
        int atom_id = groups[grouping_method_].cpu_contents[group_offset + j];
        group_per_atom_cpu_[atom_id] = i;
      }
    }
    group_per_atom_gpu_.copy_from_host(group_per_atom_cpu_.data());
    
  } else {
    num_atoms_per_group_.resize(1);
    num_atoms_ = groups[grouping_method_].cpu_size[group_id_];
    num_groups_ = 1;
    num_atoms_per_group_[0] = num_atoms_;
  }
  
  dt_in_natural_units_ = time_step * sample_interval_;
  dt_in_ps_ = dt_in_natural_units_ * TIME_UNIT_CONVERSION / 1000.0;
  x_.resize(num_atoms_ * num_correlation_steps_);
  y_.resize(num_atoms_ * num_correlation_steps_);
  z_.resize(num_atoms_ * num_correlation_steps_);
  msdx_.resize(num_correlation_steps_ * num_groups_, 0.0, Memory_Type::managed);
  msdy_.resize(num_correlation_steps_ * num_groups_, 0.0, Memory_Type::managed);
  msdz_.resize(num_correlation_steps_ * num_groups_, 0.0, Memory_Type::managed);
  msdx_out_.resize(num_correlation_steps_ * num_groups_, 0.0, Memory_Type::managed);
  msdy_out_.resize(num_correlation_steps_ * num_groups_, 0.0, Memory_Type::managed);
  msdz_out_.resize(num_correlation_steps_ * num_groups_, 0.0, Memory_Type::managed);

  num_time_origins_ = 0;
}

void MSD::process(
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
  if ((step + 1) % sample_interval_ != 0)
    return;

  const int sample_step = step / sample_interval_;
  const int correlation_step = sample_step % num_correlation_steps_;
  const int step_offset = correlation_step * num_atoms_;
  const int number_of_atoms_total = atom.number_of_atoms;

  // copy the position data at the current step to appropriate place
  if (grouping_method_ < 0 || msd_over_all_groups_) {
    gpu_copy_position<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_,
      atom.unwrapped_position.data(),
      atom.unwrapped_position.data() + number_of_atoms_total,
      atom.unwrapped_position.data() + 2 * number_of_atoms_total,
      x_.data() + step_offset,
      y_.data() + step_offset,
      z_.data() + step_offset);
  } else {
    const int group_offset = groups[grouping_method_].cpu_size_sum[group_id_];
    gpu_copy_position<<<(num_atoms_ - 1) / 128 + 1, 128>>>(
      num_atoms_,
      group_offset,
      groups[grouping_method_].contents.data(),
      atom.unwrapped_position.data(),
      atom.unwrapped_position.data() + number_of_atoms_total,
      atom.unwrapped_position.data() + 2 * number_of_atoms_total,
      x_.data() + step_offset,
      y_.data() + step_offset,
      z_.data() + step_offset);
  }
  GPU_CHECK_KERNEL

  // start to calculate the MSD when we have enough frames
  if (sample_step >= num_correlation_steps_ - 1) {
    ++num_time_origins_;

    if (msd_over_all_groups_) {
      gpu_find_msd_per_group<<<num_correlation_steps_, 128>>>(
        num_atoms_,
        num_groups_,
        correlation_step,
        num_correlation_steps_,
        group_per_atom_gpu_.data(),
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

    } else {
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
    if (save_output_every_ > 0) {
      if (0 == (step + 1) % save_output_every_) {
        std::string filename("msd_step" + std::to_string(step + 1) + ".out");
        write(filename.c_str());
      }
    }
  }
}


void MSD::write(const char* filename)
{
  CHECK(gpuDeviceSynchronize()); // needed for pre-Pascal GPU
    
  std::vector<double> sdc_x(num_correlation_steps_ * num_groups_, 0.0);
  std::vector<double> sdc_y(num_correlation_steps_ * num_groups_, 0.0);
  std::vector<double> sdc_z(num_correlation_steps_ * num_groups_, 0.0);

  // normalize by the number of atoms and number of time origins
  for (int group_id=0; group_id < num_groups_; group_id++) {
    int num_atoms = num_atoms_per_group_[group_id];
    
    // This is the case for empty groups and if the msd has yet to be computed
    double msd_scaler = 0.0;
    if (num_atoms > 0 && num_time_origins_ > 0) {
      // num_time_origins_ should be different for each nc
      msd_scaler = 1.0 / ((double)num_atoms * (double)num_time_origins_);
    } 

    int group_index = group_id * num_correlation_steps_;
    for (int nc = group_index + 0; nc < group_index + num_correlation_steps_; nc++) {
      msdx_out_[nc] = msdx_[nc] * msd_scaler;
      msdy_out_[nc] = msdy_[nc] * msd_scaler;
      msdz_out_[nc] = msdz_[nc] * msd_scaler;
    }

    const double dt2inv = 0.5 / dt_in_natural_units_;
    for (int nc = group_index + 1; nc < group_index + num_correlation_steps_; nc++) {
      sdc_x[nc] = (msdx_out_[nc] - msdx_out_[nc - 1]) * dt2inv;
      sdc_y[nc] = (msdy_out_[nc] - msdy_out_[nc - 1]) * dt2inv;
      sdc_z[nc] = (msdz_out_[nc] - msdz_out_[nc - 1]) * dt2inv;
    }
  }

  const double sdc_unit_conversion = 1.0e3 / TIME_UNIT_CONVERSION;

  FILE* fid = fopen(filename, "a");
  for (int nc = 0; nc < num_correlation_steps_; nc++) {
    fprintf(fid, "%g", nc * dt_in_ps_);
    for (int group_id = 0; group_id < num_groups_; group_id++) {
      int group_index = group_id * num_correlation_steps_;
      fprintf(
        fid,
        "% g %g %g %g %g %g",
        msdx_out_[group_index + nc],
        msdy_out_[group_index + nc],
        msdz_out_[group_index + nc],
        sdc_x[group_index + nc] * sdc_unit_conversion,
        sdc_y[group_index + nc] * sdc_unit_conversion,
        sdc_z[group_index + nc] * sdc_unit_conversion);
    }
    fprintf(fid, "\n");
  }
  fflush(fid);
  fclose(fid);
}



void MSD::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute_)
    return;
  write("msd.out");
  compute_ = false;
  grouping_method_ = -1;
}

MSD::MSD(const char** param, const int num_param, const std::vector<Group>& groups, Atom& atom)
{
  parse(param, num_param, groups);
  if (atom.unwrapped_position.size() < atom.number_of_atoms * 3) {
    atom.unwrapped_position.resize(atom.number_of_atoms * 3);
    atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
  }
  if (atom.position_temp.size() < atom.number_of_atoms * 3) {
    atom.position_temp.resize(atom.number_of_atoms * 3);
  }
  property_name = "compute_msd";
}

void MSD::parse(const char** param, const int num_param, const std::vector<Group>& groups)
{
  printf("Compute mean square displacement (MSD).\n");
  compute_ = true;

  if (num_param < 3) {
    PRINT_INPUT_ERROR("compute_msd should have at least 2 parameters.\n");
  }
  if (num_param > 8) {
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

  for (int k = 3; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, num_param, false, groups, k, grouping_method_, group_id_);

    } else if (strcmp(param[k], "all_groups") == 0) {
      msd_over_all_groups_ = true;
      // Compute MSD individually for all groups
     if (!is_valid_int(param[4], &grouping_method_)) {
        PRINT_INPUT_ERROR("Grouping method should be an integer.\n");
      }
      if (grouping_method_ < 0) {
        PRINT_INPUT_ERROR("Grouping method should >= 0.");
      }
      if (grouping_method_ >= groups.size()) {
        PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
      }
      printf("    will compute MSD for all groups in grouping %d.\n", grouping_method_);
      k += 1; // update index for next command
    } else if (strcmp(param[k], "save_every") == 0) {
      if (!is_valid_int(param[k+1], &save_output_every_)) {
        PRINT_INPUT_ERROR("save_every should be an integer.\n");
      }
      printf("    will save a copy of the MSD every %d steps.\n", save_output_every_);
      k += 1; // update index for next command
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in compute_msd.\n");
    }
  }
  if (msd_over_all_groups_ && grouping_method_ >= 0 && group_id_ >= 0) {
    PRINT_INPUT_ERROR("Cannot compute MSD over a single group and all groups at the same time");
  }
}
