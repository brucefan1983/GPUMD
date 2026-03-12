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

/*----------------------------------------------------------------------------80
Compute chunk-averaged properties using coordinate-based spatial binning.
Syntax:
  compute_chunk <sample_interval> <output_interval>
    bin/1d <dim> <origin> <delta>
    temperature [density/number] [density/mass] [vx] [vy] [vz] [fx] [fy] [fz]
Output: compute_chunk.out
------------------------------------------------------------------------------*/

#include "compute_chunk.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <cmath>

#define DIM 3
#define K_B_CHUNK 8.617333262e-5

// GPU kernel: assign atoms to 1D bins
static __global__ void gpu_assign_bins_1d(
  int N, const double* x, const double* y, const double* z,
  int axis, double origin, double invdelta, int nlayers, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double pos = (axis == 0) ? x[n] : ((axis == 1) ? y[n] : z[n]);
    int bin_id = static_cast<int>((pos - origin) * invdelta);
    if (bin_id < 0) bin_id = 0;
    if (bin_id >= nlayers) bin_id = nlayers - 1;
    ichunk[n] = bin_id;
  }
}

// GPU kernel: assign atoms to 2D bins
static __global__ void gpu_assign_bins_2d(
  int N, const double* x, const double* y, const double* z,
  int axis0, int axis1,
  double origin0, double origin1,
  double invdelta0, double invdelta1,
  int nlayers0, int nlayers1, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double pos0 = (axis0 == 0) ? x[n] : ((axis0 == 1) ? y[n] : z[n]);
    double pos1 = (axis1 == 0) ? x[n] : ((axis1 == 1) ? y[n] : z[n]);
    int b0 = static_cast<int>((pos0 - origin0) * invdelta0);
    int b1 = static_cast<int>((pos1 - origin1) * invdelta1);
    if (b0 < 0) b0 = 0; if (b0 >= nlayers0) b0 = nlayers0 - 1;
    if (b1 < 0) b1 = 0; if (b1 >= nlayers1) b1 = nlayers1 - 1;
    ichunk[n] = b0 + nlayers0 * b1;
  }
}

// GPU kernel: assign atoms to 3D bins
static __global__ void gpu_assign_bins_3d(
  int N, const double* x, const double* y, const double* z,
  int axis0, int axis1, int axis2,
  double o0, double o1, double o2,
  double id0, double id1, double id2,
  int n0, int n1, int n2, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double pos[3] = {x[n], y[n], z[n]};
    int b0 = static_cast<int>((pos[axis0] - o0) * id0);
    int b1 = static_cast<int>((pos[axis1] - o1) * id1);
    int b2 = static_cast<int>((pos[axis2] - o2) * id2);
    if (b0 < 0) b0 = 0; if (b0 >= n0) b0 = n0 - 1;
    if (b1 < 0) b1 = 0; if (b1 >= n1) b1 = n1 - 1;
    if (b2 < 0) b2 = 0; if (b2 >= n2) b2 = n2 - 1;
    ichunk[n] = b0 + n0 * (b1 + n1 * b2);
  }
}

// GPU kernel: count atoms per chunk
static __global__ void gpu_count_per_chunk(
  int N, const int* ichunk, int nchunk, int* count)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk) atomicAdd(&count[c], 1);
  }
}

// GPU kernel: sum scalar property per chunk
static __global__ void gpu_sum_scalar_per_chunk(
  int N, const int* ichunk, const double* prop,
  int nchunk, int offset, int nvalues, double* out)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk)
      atomicAdd(&out[c * nvalues + offset], prop[n]);
  }
}

// GPU kernel: sum kinetic energy per chunk (for temperature)
static __global__ void gpu_sum_ke_per_chunk(
  int N, const int* ichunk, const double* mass,
  const double* vx, const double* vy, const double* vz,
  int nchunk, int offset, int nvalues, double* out)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk) {
      double v2 = vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n];
      atomicAdd(&out[c * nvalues + offset], 0.5 * mass[n] * v2);
    }
  }
}

ComputeChunk::ComputeChunk(const char** param, int num_param, Box& box)
{
  property_name = "compute_chunk";
  for (int i = 0; i < 3; i++) {
    axis_[i] = 0; origin_[i] = 0.0;
    delta_[i] = 1.0; invdelta_[i] = 1.0; nlayers_[i] = 1;
  }
  parse(param, num_param, box);
}

void ComputeChunk::parse(const char** param, int num_param, Box& box)
{
  printf("Compute chunk-averaged properties:\n");

  if (num_param < 6) {
    PRINT_INPUT_ERROR("compute_chunk requires at least 5 parameters.\n");
  }

  // param[1]: sample_interval
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample_interval should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample_interval should > 0.\n");
  }

  // param[2]: output_interval
  if (!is_valid_int(param[2], &output_interval_)) {
    PRINT_INPUT_ERROR("output_interval should be an integer.\n");
  }
  if (output_interval_ <= 0) {
    PRINT_INPUT_ERROR("output_interval should > 0.\n");
  }

  // param[3]: bin style -> parse bin params, returns index of next param
  int next = parse_bin_params(param, num_param, 3, box);

  // remaining params are properties
  if (next >= num_param) {
    PRINT_INPUT_ERROR("compute_chunk requires at least one property.\n");
  }

  for (int k = next; k < num_param; ++k) {
    if (strcmp(param[k], "temperature") == 0) {
      compute_temperature_ = 1;
      number_of_scalars_++;
      printf("    temperature\n");
    } else if (strcmp(param[k], "density/number") == 0) {
      compute_density_number_ = 1;
      number_of_scalars_++;
      printf("    density/number\n");
    } else if (strcmp(param[k], "density/mass") == 0) {
      compute_density_mass_ = 1;
      number_of_scalars_++;
      printf("    density/mass\n");
    } else if (strcmp(param[k], "vx") == 0) {
      compute_vx_ = 1; number_of_scalars_++; printf("    vx\n");
    } else if (strcmp(param[k], "vy") == 0) {
      compute_vy_ = 1; number_of_scalars_++; printf("    vy\n");
    } else if (strcmp(param[k], "vz") == 0) {
      compute_vz_ = 1; number_of_scalars_++; printf("    vz\n");
    } else if (strcmp(param[k], "fx") == 0) {
      compute_fx_ = 1; number_of_scalars_++; printf("    fx\n");
    } else if (strcmp(param[k], "fy") == 0) {
      compute_fy_ = 1; number_of_scalars_++; printf("    fy\n");
    } else if (strcmp(param[k], "fz") == 0) {
      compute_fz_ = 1; number_of_scalars_++; printf("    fz\n");
    } else {
      PRINT_INPUT_ERROR("Invalid property for compute_chunk.\n");
    }
  }

  printf("    with %dD binning, %d chunks.\n", dim_, nchunk_);
  printf("    with sampling interval %d.\n", sample_interval_);
  printf("    and output interval %d.\n", output_interval_);
}

static int parse_one_axis(
  const char** param, int idx, int* axis, double* origin, double* delta,
  double* invdelta, int* nlayers, double* box_len_out, const Box& box)
{
  // Use geometric thickness (volume/area) which is correct for triclinic boxes
  double vol = box.get_volume();
  double box_lengths[3] = {
    vol / box.get_area(0),  // thickness along x
    vol / box.get_area(1),  // thickness along y
    vol / box.get_area(2)   // thickness along z
  };

  if (strcmp(param[idx], "x") == 0) *axis = 0;
  else if (strcmp(param[idx], "y") == 0) *axis = 1;
  else if (strcmp(param[idx], "z") == 0) *axis = 2;
  else { PRINT_INPUT_ERROR("dim must be x, y, or z.\n"); }

  double box_len = box_lengths[*axis];

  // GPUMD wraps positions to [0, L], bins always start from 0.
  if (strcmp(param[idx + 1], "lower") == 0) {
    *origin = 0.0;
  } else {
    PRINT_INPUT_ERROR("origin must be lower.\n");
  }

  if (!is_valid_real(param[idx + 2], delta)) {
    PRINT_INPUT_ERROR("delta must be a positive number.\n");
  }
  if (*delta <= 0.0) {
    PRINT_INPUT_ERROR("delta must be positive.\n");
  }
  *invdelta = 1.0 / (*delta);
  // Use ceil so the last bin is not wider than delta
  *nlayers = static_cast<int>(ceil(box_len / (*delta)));
  if (*nlayers <= 0) *nlayers = 1;
  *box_len_out = box_len;

  return idx + 3;
}

int ComputeChunk::parse_bin_params(
  const char** param, int num_param, int start, Box& box)
{
  if (strcmp(param[start], "bin/1d") == 0) {
    dim_ = 1; ncoord_ = 1;
    if (start + 4 > num_param) {
      PRINT_INPUT_ERROR("bin/1d requires: dim origin delta.\n");
    }
    int next = parse_one_axis(
      param, start + 1, &axis_[0], &origin_[0], &delta_[0],
      &invdelta_[0], &nlayers_[0], &box_length_[0], box);
    nchunk_ = nlayers_[0];
    return next;
  } else if (strcmp(param[start], "bin/2d") == 0) {
    dim_ = 2; ncoord_ = 2;
    if (start + 7 > num_param) {
      PRINT_INPUT_ERROR("bin/2d requires: dim origin delta dim origin delta.\n");
    }
    int next = parse_one_axis(
      param, start + 1, &axis_[0], &origin_[0], &delta_[0],
      &invdelta_[0], &nlayers_[0], &box_length_[0], box);
    next = parse_one_axis(
      param, next, &axis_[1], &origin_[1], &delta_[1],
      &invdelta_[1], &nlayers_[1], &box_length_[1], box);
    if (axis_[0] == axis_[1]) {
      PRINT_INPUT_ERROR("bin/2d requires two different axes.\n");
    }
    nchunk_ = nlayers_[0] * nlayers_[1];
    return next;
  } else if (strcmp(param[start], "bin/3d") == 0) {
    dim_ = 3; ncoord_ = 3;
    if (start + 10 > num_param) {
      PRINT_INPUT_ERROR("bin/3d requires: dim origin delta (x3).\n");
    }
    int next = parse_one_axis(
      param, start + 1, &axis_[0], &origin_[0], &delta_[0],
      &invdelta_[0], &nlayers_[0], &box_length_[0], box);
    next = parse_one_axis(
      param, next, &axis_[1], &origin_[1], &delta_[1],
      &invdelta_[1], &nlayers_[1], &box_length_[1], box);
    next = parse_one_axis(
      param, next, &axis_[2], &origin_[2], &delta_[2],
      &invdelta_[2], &nlayers_[2], &box_length_[2], box);
    if (axis_[0] == axis_[1] || axis_[0] == axis_[2] || axis_[1] == axis_[2]) {
      PRINT_INPUT_ERROR("bin/3d requires three different axes.\n");
    }
    nchunk_ = nlayers_[0] * nlayers_[1] * nlayers_[2];
    return next;
  } else {
    PRINT_INPUT_ERROR("bin style must be bin/1d, bin/2d, or bin/3d.\n");
    return start;
  }
}

void ComputeChunk::preprocess(
  const int number_of_steps, const double time_step,
  Integrate& integrate, std::vector<Group>& group,
  Atom& atom, Box& box, Force& force)
{
  if (number_of_scalars_ == 0) return;

  ichunk_.resize(atom.number_of_atoms);
  gpu_count_.resize(nchunk_);
  gpu_values_.resize(nchunk_ * number_of_scalars_);
  cpu_count_sum_.resize(nchunk_, 0);
  cpu_values_sum_.resize(nchunk_ * number_of_scalars_, 0.0);
  num_samples_ = 0;

  calculate_chunk_volumes(box);
  calculate_chunk_coords(box);

  fid_ = my_fopen("compute_chunk.out", "a");
}

void ComputeChunk::process(
  const int number_of_steps, int step,
  const int fixed_group, const int move_group,
  const double global_time, const double temperature,
  Integrate& integrate, Box& box, std::vector<Group>& group,
  GPU_Vector<double>& thermo, Atom& atom, Force& force)
{
  if (number_of_scalars_ == 0) return;
  if ((step + 1) % sample_interval_ != 0) return;

  assign_chunks(atom, box);
  sample(atom);
  num_samples_++;

  // output_interval_ means "output every N samples" (same as compute command)
  if (num_samples_ % output_interval_ == 0) {
    output_results(step + 1);
    std::fill(cpu_count_sum_.begin(), cpu_count_sum_.end(), 0);
    std::fill(cpu_values_sum_.begin(), cpu_values_sum_.end(), 0.0);
    num_samples_ = 0;
  }
}

void ComputeChunk::postprocess(
  Atom& atom, Box& box, Integrate& integrate,
  const int number_of_steps, const double time_step, const double temperature)
{
  if (number_of_scalars_ == 0) return;
  if (fid_) fclose(fid_);
  fid_ = nullptr;

  compute_temperature_ = 0;
  compute_density_number_ = 0;
  compute_density_mass_ = 0;
  compute_vx_ = compute_vy_ = compute_vz_ = 0;
  compute_fx_ = compute_fy_ = compute_fz_ = 0;
  number_of_scalars_ = 0;
}

void ComputeChunk::assign_chunks(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  const double* x = atom.position_per_atom.data();
  const double* y = x + N;
  const double* z = y + N;

  if (dim_ == 1) {
    gpu_assign_bins_1d<<<grid_size, block_size>>>(
      N, x, y, z, axis_[0], origin_[0], invdelta_[0], nlayers_[0],
      ichunk_.data());
  } else if (dim_ == 2) {
    gpu_assign_bins_2d<<<grid_size, block_size>>>(
      N, x, y, z, axis_[0], axis_[1],
      origin_[0], origin_[1], invdelta_[0], invdelta_[1],
      nlayers_[0], nlayers_[1], ichunk_.data());
  } else {
    gpu_assign_bins_3d<<<grid_size, block_size>>>(
      N, x, y, z,
      axis_[0], axis_[1], axis_[2],
      origin_[0], origin_[1], origin_[2],
      invdelta_[0], invdelta_[1], invdelta_[2],
      nlayers_[0], nlayers_[1], nlayers_[2], ichunk_.data());
  }
  GPU_CHECK_KERNEL
}

void ComputeChunk::sample(const Atom& atom)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;

  gpu_count_.fill(0);
  gpu_values_.fill(0.0);

  gpu_count_per_chunk<<<grid_size, block_size>>>(
    N, ichunk_.data(), nchunk_, gpu_count_.data());
  GPU_CHECK_KERNEL

  const double* mass = atom.mass.data();
  const double* vx = atom.velocity_per_atom.data();
  const double* vy = vx + N;
  const double* vz = vy + N;
  const double* fx = atom.force_per_atom.data();
  const double* fy = fx + N;
  const double* fz = fy + N;

  int idx = 0;
  if (compute_temperature_) {
    gpu_sum_ke_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), mass, vx, vy, vz,
      nchunk_, idx, number_of_scalars_, gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_density_number_) { idx++; } // count-based, no kernel needed
  if (compute_density_mass_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), mass, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_vx_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), vx, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_vy_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), vy, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_vz_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), vz, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_fx_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), fx, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_fy_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), fy, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_fz_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), fz, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }

  // accumulate to CPU
  std::vector<int> cpu_count(nchunk_);
  std::vector<double> cpu_values(nchunk_ * number_of_scalars_);
  gpu_count_.copy_to_host(cpu_count.data(), nchunk_);
  gpu_values_.copy_to_host(cpu_values.data(), nchunk_ * number_of_scalars_);
  for (int i = 0; i < nchunk_; i++) cpu_count_sum_[i] += cpu_count[i];
  for (int i = 0; i < nchunk_ * number_of_scalars_; i++)
    cpu_values_sum_[i] += cpu_values[i];
}

void ComputeChunk::output_results(int step)
{
  if (num_samples_ == 0) return;

  for (int chunk = 0; chunk < nchunk_; chunk++) {
    double count = static_cast<double>(cpu_count_sum_[chunk]) / num_samples_;

    // write: chunk_id coord(s)
    fprintf(fid_, "%d ", chunk);
    for (int c = 0; c < ncoord_; c++)
      fprintf(fid_, "%.6f ", chunk_coords_cpu_[chunk * ncoord_ + c]);
    fprintf(fid_, "%.1f ", count);

    int idx = 0;
    if (compute_temperature_) {
      double ke = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      double temp = (count > 0) ? (2.0 * ke / (K_B_CHUNK * DIM * count)) : 0.0;
      fprintf(fid_, "%.10e ", temp);
      idx++;
    }
    if (compute_density_number_) {
      double dn = count / chunk_volume_cpu_[chunk];
      fprintf(fid_, "%.10e ", dn);
      idx++;
    }
    if (compute_density_mass_) {
      double dm = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      dm /= chunk_volume_cpu_[chunk];
      fprintf(fid_, "%.10e ", dm);
      idx++;
    }
    if (compute_vx_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_vy_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_vz_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_fx_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_fy_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_fz_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    fprintf(fid_, "\n");
  }
  fflush(fid_);
}

void ComputeChunk::calculate_chunk_volumes(const Box& box)
{
  chunk_volume_cpu_.resize(nchunk_);
  double box_vol = box.get_volume();

  // Helper: actual width of bin i along dimension d
  // Last bin may be narrower than delta when L is not a multiple of delta
  auto bin_width = [&](int d, int i) -> double {
    double remainder = box_length_[d] - (nlayers_[d] - 1) * delta_[d];
    return (i < nlayers_[d] - 1) ? delta_[d] : remainder;
  };

  if (dim_ == 1) {
    double cross_area = box_vol / box_length_[0];
    for (int i = 0; i < nchunk_; i++)
      chunk_volume_cpu_[i] = cross_area * bin_width(0, i);
  } else if (dim_ == 2) {
    int third = 3 - axis_[0] - axis_[1];
    double thickness = box_vol / box.get_area(third);
    int idx = 0;
    for (int j = 0; j < nlayers_[1]; j++)
      for (int i = 0; i < nlayers_[0]; i++) {
        chunk_volume_cpu_[idx] = bin_width(0, i) * bin_width(1, j) * thickness;
        idx++;
      }
  } else {
    int idx = 0;
    for (int k = 0; k < nlayers_[2]; k++)
      for (int j = 0; j < nlayers_[1]; j++)
        for (int i = 0; i < nlayers_[0]; i++) {
          chunk_volume_cpu_[idx] = bin_width(0, i) * bin_width(1, j) * bin_width(2, k);
          idx++;
        }
  }
}

void ComputeChunk::calculate_chunk_coords(const Box& box)
{
  chunk_coords_cpu_.resize(nchunk_ * ncoord_);

  // Bin center: for normal bins it's origin + (i+0.5)*delta,
  // for the last bin it's origin + (nlayers-1)*delta + remainder/2.
  auto bin_center = [&](int d, int i) -> double {
    if (i < nlayers_[d] - 1) {
      return origin_[d] + (i + 0.5) * delta_[d];
    } else {
      double remainder = box_length_[d] - (nlayers_[d] - 1) * delta_[d];
      return origin_[d] + (nlayers_[d] - 1) * delta_[d] + remainder * 0.5;
    }
  };

  if (dim_ == 1) {
    for (int i = 0; i < nchunk_; i++)
      chunk_coords_cpu_[i] = bin_center(0, i);
  } else if (dim_ == 2) {
    int idx = 0;
    for (int j = 0; j < nlayers_[1]; j++)
      for (int i = 0; i < nlayers_[0]; i++) {
        chunk_coords_cpu_[idx * 2 + 0] = bin_center(0, i);
        chunk_coords_cpu_[idx * 2 + 1] = bin_center(1, j);
        idx++;
      }
  } else {
    int idx = 0;
    for (int k = 0; k < nlayers_[2]; k++)
      for (int j = 0; j < nlayers_[1]; j++)
        for (int i = 0; i < nlayers_[0]; i++) {
          chunk_coords_cpu_[idx * 3 + 0] = bin_center(0, i);
          chunk_coords_cpu_[idx * 3 + 1] = bin_center(1, j);
          chunk_coords_cpu_[idx * 3 + 2] = bin_center(2, k);
          idx++;
        }
  }
}
