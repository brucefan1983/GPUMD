/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Compute chunk-averaged properties using coordinate-based spatial binning.

Output: compute_chunk.out

Major contributors:
  Wenhao Luo: Initial implementation.
------------------------------------------------------------------------------*/

#include "compute_chunk.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <climits>
#include <cmath>
#include <iterator>

#define DIM 3
#define K_B_CHUNK 8.617333262e-5

namespace
{
struct BinAxis {
  double inverse_row[3];
  double bins_per_fraction;
  int num_bins;
  int periodic;
};

struct BinAxes {
  BinAxis axes[3];
};
} // namespace

static __global__ void
gpu_assign_bins(int N, const double* position, int dim, BinAxes grid, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  const double x = position[n], y = position[n + N], z = position[n + 2 * N];
  int chunk = 0;
  for (int d = dim - 1; d >= 0; --d) {
    const BinAxis& axis = grid.axes[d];
    const double* row = axis.inverse_row;
    double s = row[0] * x + row[1] * y + row[2] * z;
    if (axis.periodic)
      s -= floor(s);
    double bin = fmin(fmax(s * axis.bins_per_fraction, 0.0), axis.num_bins - 1.0);
    chunk = chunk * axis.num_bins + static_cast<int>(bin);
  }
  ichunk[n] = chunk;
}

// GPU kernel: count atoms per chunk
static __global__ void gpu_count_per_chunk(int N, const int* ichunk, int nchunk, int* count)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk)
      atomicAdd(&count[c], 1);
  }
}

// GPU kernel: sum scalar property per chunk
static __global__ void gpu_sum_scalar_per_chunk(
  int N, const int* ichunk, const double* prop, int nchunk, int offset, int nvalues, double* out)
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
  int N,
  const int* ichunk,
  const double* mass,
  const double* vx,
  const double* vy,
  const double* vz,
  int nchunk,
  int offset,
  int nvalues,
  double* out)
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

ComputeChunk::ComputeChunk(const std::vector<std::string>& tokens, Box& box)
{
  action_name = "compute_chunk";
  printf("Compute chunk-averaged properties:\n");
  if (tokens.size() < 4) {
    PRINT_INPUT_ERROR("compute_chunk requires sampling intervals and a bin style.\n");
  }
  // param[1]: sample_interval
  if (!is_valid_int(tokens[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample_interval should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample_interval should > 0.\n");
  }

  // param[2]: output_interval
  if (!is_valid_int(tokens[2], &output_interval_)) {
    PRINT_INPUT_ERROR("output_interval should be an integer.\n");
  }
  if (output_interval_ <= 0) {
    PRINT_INPUT_ERROR("output_interval should > 0.\n");
  }

  if (tokens[3] == "bin/1d") {
    dim_ = 1;
  } else if (tokens[3] == "bin/2d") {
    dim_ = 2;
  } else if (tokens[3] == "bin/3d") {
    dim_ = 3;
  } else {
    PRINT_INPUT_ERROR("bin style must be bin/1d, bin/2d, or bin/3d.\n");
  }
  const int first_quantity = 4 + 3 * dim_;
  if (tokens.size() <= first_quantity) {
    PRINT_INPUT_ERROR(
      "compute_chunk requires dim origin delta per axis and at least one property.\n");
  }
  const double volume = box.get_volume();
  for (int d = 0; d < dim_; ++d) {
    const int i = 4 + 3 * d;
    const std::string& axis = tokens[i];
    if (axis != "x" && axis != "y" && axis != "z") {
      PRINT_INPUT_ERROR("dim must be x, y, or z.\n");
    }
    axis_[d] = axis[0] - 'x';
    if (std::find(axis_, axis_ + d, axis_[d]) != axis_ + d) {
      PRINT_INPUT_ERROR("compute_chunk requires distinct binning axes.\n");
    }
    if (tokens[i + 1] != "lower") {
      PRINT_INPUT_ERROR("origin must be lower.\n");
    }
    double delta;
    if (!is_valid_real(tokens[i + 2], &delta) || delta <= 0.0) {
      PRINT_INPUT_ERROR("delta must be a finite positive number.\n");
    }
    const double thickness = volume / box.get_area(axis_[d]);
    bins_per_fraction_[d] = std::max(1.0, thickness / delta);
    const double layers = std::ceil(bins_per_fraction_[d]);
    if (layers > INT_MAX / nchunk_) {
      PRINT_INPUT_ERROR("Too many bins for compute_chunk.\n");
    }
    nlayers_[d] = static_cast<int>(layers);
    nchunk_ *= nlayers_[d];
  }

  const std::string names[] = {
    "temperature", "density/number", "density/mass", "vx", "vy", "vz", "fx", "fy", "fz"};
  for (size_t i = first_quantity; i < tokens.size(); ++i) {
    const auto name = std::find(std::begin(names), std::end(names), tokens[i]);
    if (name == std::end(names)) {
      PRINT_INPUT_ERROR("Invalid property for compute_chunk.\n");
    }
    const Quantity quantity = static_cast<Quantity>(name - names);
    if (std::find(quantities_.begin(), quantities_.end(), quantity) != quantities_.end()) {
      PRINT_INPUT_ERROR("Duplicate property for compute_chunk.\n");
    }
    quantities_.push_back(quantity);
    printf("    %s\n", tokens[i].c_str());
  }
  if (nchunk_ > INT_MAX / quantities_.size()) {
    PRINT_INPUT_ERROR("Too many values for compute_chunk.\n");
  }
  printf("    with %dD binning, %d chunks.\n", dim_, nchunk_);
  printf("    with sampling interval %d.\n", sample_interval_);
  printf("    and output interval %d.\n", output_interval_);
}

void ComputeChunk::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  const size_t nvalues = nchunk_ * quantities_.size();
  ichunk_.resize(atom.number_of_atoms);
  gpu_count_.resize(nchunk_);
  gpu_values_.resize(nvalues);
  cpu_count_.resize(nchunk_);
  cpu_values_.resize(nvalues);
  count_sum_.assign(nchunk_, 0);
  values_sum_.assign(nvalues, 0.0);

  volume_fraction_.assign(nchunk_, 1.0);
  center_fraction_.resize(static_cast<size_t>(nchunk_) * dim_);
  for (int c = 0; c < nchunk_; ++c) {
    int index = c;
    for (int d = 0; d < dim_; ++d) {
      const int layer = index % nlayers_[d];
      index /= nlayers_[d];
      const double lower = layer / bins_per_fraction_[d];
      const double upper = (layer == nlayers_[d] - 1) ? 1.0 : (layer + 1) / bins_per_fraction_[d];
      volume_fraction_[c] *= upper - lower;
      center_fraction_[static_cast<size_t>(c) * dim_ + d] = (lower + upper) * 0.5;
    }
  }
  fid_ = my_fopen("compute_chunk.out", "a");
}

void ComputeChunk::end_of_step(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % sample_interval_ != 0)
    return;
  sample(atom, box);
  if (++num_samples_ == output_interval_) {
    output_results();
    std::fill(count_sum_.begin(), count_sum_.end(), 0);
    std::fill(values_sum_.begin(), values_sum_.end(), 0.0);
    std::fill_n(thickness_sum_, dim_, 0.0);
    num_samples_ = 0;
  }
}

void ComputeChunk::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(fid_);
}

void ComputeChunk::sample(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  const double volume = box.get_volume();
  const int periodic[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  BinAxes grid = {};
  for (int d = 0; d < dim_; ++d) {
    const int axis = axis_[d];
    std::copy_n(box.cpu_h + 9 + 3 * axis, 3, grid.axes[d].inverse_row);
    grid.axes[d].periodic = periodic[axis];
    grid.axes[d].bins_per_fraction = bins_per_fraction_[d];
    grid.axes[d].num_bins = nlayers_[d];
    thickness_sum_[d] += volume / box.get_area(axis);
  }
  gpu_assign_bins<<<grid_size, block_size>>>(
    N, atom.position_per_atom.data(), dim_, grid, ichunk_.data());
  GPU_CHECK_KERNEL
  gpu_count_.fill(0);
  gpu_values_.fill(0.0);
  gpu_count_per_chunk<<<grid_size, block_size>>>(N, ichunk_.data(), nchunk_, gpu_count_.data());
  GPU_CHECK_KERNEL

  const int nvalues = quantities_.size();
  const double* mass = atom.mass.data();
  const double* velocity = atom.velocity_per_atom.data();
  for (int q = 0; q < nvalues; ++q) {
    const Quantity quantity = quantities_[q];
    if (quantity == NumberDensity)
      continue;
    if (quantity == Temperature) {
      gpu_sum_ke_per_chunk<<<grid_size, block_size>>>(
        N,
        ichunk_.data(),
        mass,
        velocity,
        velocity + N,
        velocity + 2 * N,
        nchunk_,
        q,
        nvalues,
        gpu_values_.data());
    } else {
      const double* prop = mass;
      if (quantity >= Fx) {
        prop = atom.force_per_atom.data() + (quantity - Fx) * N;
      } else if (quantity >= Vx) {
        prop = velocity + (quantity - Vx) * N;
      }
      gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
        N, ichunk_.data(), prop, nchunk_, q, nvalues, gpu_values_.data());
    }
    GPU_CHECK_KERNEL
  }
  gpu_count_.copy_to_host(cpu_count_.data());
  gpu_values_.copy_to_host(cpu_values_.data());
  for (int c = 0; c < nchunk_; ++c) {
    count_sum_[c] += cpu_count_[c];
    const double bin_volume = volume * volume_fraction_[c];
    for (int q = 0; q < nvalues; ++q) {
      const int i = c * nvalues + q;
      double value = cpu_values_[i];
      if (quantities_[q] == NumberDensity)
        value = cpu_count_[c] / bin_volume;
      else if (quantities_[q] == MassDensity)
        value /= bin_volume;
      values_sum_[i] += value;
    }
  }
}

void ComputeChunk::output_results()
{
  for (int c = 0; c < nchunk_; ++c) {
    fprintf(fid_, "%d ", c);
    for (int d = 0; d < dim_; ++d) {
      const double center =
        center_fraction_[static_cast<size_t>(c) * dim_ + d] * (thickness_sum_[d] / num_samples_);
      fprintf(fid_, "%.6f ", center);
    }
    const double count = count_sum_[c];
    fprintf(fid_, "%.1f ", count / num_samples_);
    for (size_t q = 0; q < quantities_.size(); ++q) {
      const Quantity quantity = quantities_[q];
      double value = values_sum_[c * quantities_.size() + q];
      if (quantity == NumberDensity || quantity == MassDensity) {
        value /= num_samples_;
      } else if (count > 0.0) {
        value /= count;
        if (quantity == Temperature)
          value *= 2.0 / (K_B_CHUNK * DIM);
        else if (quantity >= Vx && quantity <= Vz)
          value /= TIME_UNIT_CONVERSION;
      }
      fprintf(fid_, "%.10e ", value);
    }
    fprintf(fid_, "\n");
  }
  fflush(fid_);
}
