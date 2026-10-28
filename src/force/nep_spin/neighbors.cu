/* Copyright 2017 Zheyong Fan and GPUMD development team
   This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#include "nep_spin.cuh"
#include "utilities/error.cuh"
#include <algorithm>

namespace
{
// The same cell traversal as Neighbor, with checked writes and per-instance state.
static __global__ void find_spin_neighbors(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const int* __restrict__ type,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const float cutoff_square,
  const int neighbor_capacity,
  int* required_capacity)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    int z_begin, z_end;
    int y_begin, y_end;
    int x_begin, x_end;
    get_neighbor_cell_offset_range(box.pbc_z, nz, z_begin, z_end);
    get_neighbor_cell_offset_range(box.pbc_y, ny, y_begin, y_end);
    get_neighbor_cell_offset_range(box.pbc_x, nx, x_begin, x_end);

    // get radial descriptors
    for (int k = z_begin; k < z_end; ++k) {
      for (int j = y_begin; j < y_end; ++j) {
        for (int i = x_begin; i < x_end; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          else if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          else if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          else if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              float x12 = x[n2] - x1;
              float y12 = y[n2] - y1;
              float z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const float d2 = x12 * x12 + y12 * y12 + z12 * z12;

              if (d2 < cutoff_square) {
                if (count < neighbor_capacity) {
                  NL[static_cast<size_t>(N) * count + n1] = n2;
                }
                ++count;
              }
            }
          }
        }
      }
    }
    NN[n1] = min(count, neighbor_capacity);
    atomicMax(required_capacity, count);
  }
}

static __global__ void
check_displacement(int n, Box box, const double* position, const double* reference, int* changed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  double x = position[i] - reference[i];
  double y = position[n + i] - reference[n + i];
  double z = position[2 * n + i] - reference[2 * n + i];
  apply_mic(box, x, y, z);
  if (x * x + y * y + z * z > 0.25)
    atomicExch(changed, 1);
}
} // namespace

void NEP_Spin::Neighbors::initialize(double rc, int atoms, int capacity)
{
  valid_ = false;
  const double buffered = rc + skin_;
  capacity_ = capacity * buffered * buffered * buffered / (rc * rc * rc);
  if (NN.size() != atoms) {
    NN.resize(atoms);
    cell_count_.resize(atoms);
    cell_sum_.resize(atoms);
    cell_contents_.resize(atoms);
    reference_.resize(size_t(atoms) * 3);
  }
  if (NL.size() != size_t(atoms) * capacity_)
    NL.resize(size_t(atoms) * capacity_);
  if (flag_.size() != 1)
    flag_.resize(1);
}

void NEP_Spin::Neighbors::find_neighbor_global(
  double rc, Box& box, const GPU_Vector<int>& type, const GPU_Vector<double>& position)
{
  const int n = type.size();
  const int grid = (n + 255) / 256;
  int changed = !valid_ || !std::equal(box_, box_ + 9, box.cpu_h);
  if (!changed) {
    flag_.fill(0);
    check_displacement<<<grid, 256>>>(n, box, position.data(), reference_.data(), flag_.data());
    GPU_CHECK_KERNEL
    flag_.copy_to_host(&changed);
  }
  if (!changed)
    return;
  const double cutoff = rc + skin_;
  int bins[3];
  box.get_num_bins(0.5 * cutoff, bins);
  find_cell_list(0.5 * cutoff, bins, box, position, cell_count_, cell_sum_, cell_contents_);
  flag_.fill(0);
  find_spin_neighbors<<<grid, 256>>>(
    box,
    n,
    0,
    n,
    type.data(),
    cell_count_.data(),
    cell_sum_.data(),
    cell_contents_.data(),
    NN.data(),
    NL.data(),
    position.data(),
    position.data() + n,
    position.data() + 2 * n,
    bins[0],
    bins[1],
    bins[2],
    2.0 / cutoff,
    cutoff * cutoff,
    capacity_,
    flag_.data());
  GPU_CHECK_KERNEL
  int required;
  flag_.copy_to_host(&required);
  if (required > capacity_) {
    PRINT_INPUT_ERROR(
      "NEP-spin neighbor capacity is insufficient; increase the model neighbor capacity.");
  }
  gpu_sort_neighbor_list<<<n, min(1024, capacity_), capacity_ * sizeof(int)>>>(
    n, NN.data(), NL.data());
  GPU_CHECK_KERNEL
  CHECK(gpuMemcpy(
    reference_.data(), position.data(), size_t(n) * 3 * sizeof(double), gpuMemcpyDeviceToDevice));
  std::copy(box.cpu_h, box.cpu_h + 9, box_);
  valid_ = true;
}
