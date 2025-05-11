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
neighbor list.
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cstring>

static __device__ void find_cell_id(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(box, x, y, z, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);
}

static __global__ void find_cell_counts(
  const Box box,
  const int N,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents(
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

static __global__ void gpu_find_neighbor_ON1(
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
  const double cutoff_square)
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

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              if (d2 < cutoff_square) {
                NL[count++ * N + n1] = n2;
              }
            }
          }
        }
      }
    }
    NN[n1] = count;
  }
}

void find_cell_list(
  const double rc,
  const int* num_bins,
  Box& box,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int N = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  CHECK(gpuMemset(cell_count.data(), 0, sizeof(int) * N_cells));
  CHECK(gpuMemset(cell_count_sum.data(), 0, sizeof(int) * N_cells));
  CHECK(gpuMemset(cell_contents.data(), 0, sizeof(int) * N));

  find_cell_counts<<<grid_size, block_size>>>(
    box, N, cell_count.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
    thrust::device, cell_count.data(), cell_count.data() + N_cells, cell_count_sum.data());

  CHECK(gpuMemset(cell_count.data(), 0, sizeof(int) * N_cells));

  find_cell_contents<<<grid_size, block_size>>>(
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL
}

static void __global__ set_to_zero(int size, int* data)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    data[n] = 0;
  }
}

void find_cell_list(
  gpuStream_t& stream,
  const double rc,
  const int* num_bins,
  Box& box,
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int offset = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + offset;
  const double* z = position_per_atom.data() + offset * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_count_sum.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count_sum.size(), cell_count_sum.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_contents.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_contents.size(), cell_contents.data());
  GPU_CHECK_KERNEL

  find_cell_counts<<<grid_size, block_size, 0, stream>>>(
    box, N, cell_count.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
#ifdef USE_HIP
    thrust::hip::par.on(stream),
#else
    thrust::cuda::par.on(stream),
#endif
    cell_count.data(),
    cell_count.data() + N_cells,
    cell_count_sum.data());

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  find_cell_contents<<<grid_size, block_size, 0, stream>>>(
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL
}

void find_neighbor(
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}

// For ILP, the neighbor could not contain atoms in the same layer
static __global__ void gpu_find_neighbor_ON1_ilp(
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
  int* big_ilp_NN,
  int* big_ilp_NL,
  const int* group_label,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square,
  const double big_ilp_cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  int count = 0;
  int ilp_count = 0;
  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            // neighbors in different layers
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              bool different_layer = group_label[n1] != group_label[n2];
              if (different_layer && d2 < cutoff_square) {
                NL[count++ * N + n1] = n2;
              } else if (!different_layer && d2 < big_ilp_cutoff_square) {
                big_ilp_NL[ilp_count++ * N + n1] = n2;
              }

            }
          }
        }
      }
    }
    NN[n1] = count;
    big_ilp_NN[n1] = ilp_count;
  }
}

void find_neighbor_ilp(
  const int N1,
  const int N2,
  double rc,
  double big_ilp_cutoff_square,
  Box& box,
  const int* group_label,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL,
  GPU_Vector<int>& big_ilp_NN,
  GPU_Vector<int>& big_ilp_NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_ilp<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    big_ilp_NN.data(),
    big_ilp_NL.data(),
    group_label,
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc,
    big_ilp_cutoff_square);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list_ilp<<<N, min(1024, MN), MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL

  const int big_ilp_MN = big_ilp_NL.size() / big_ilp_NN.size();
  gpu_sort_neighbor_list<<<N, big_ilp_MN, big_ilp_MN * sizeof(int)>>>(N, big_ilp_NN.data(), big_ilp_NL.data());
  GPU_CHECK_KERNEL
}

static __global__ void gpu_find_neighbor_ON1_SW(
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
  const int* group_label,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square)
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

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    // get radial descriptors
    for (int k = -z_lim; k <= z_lim; ++k) {
      for (int j = -y_lim; j <= y_lim; ++j) {
        for (int i = -x_lim; i <= x_lim; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            if (n2 >= N1 && n2 < N2 && n1 != n2) {

              double x12 = x[n2] - x1;
              double y12 = y[n2] - y1;
              double z12 = z[n2] - z1;
              apply_mic(box, x12, y12, z12);
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

              if (d2 < cutoff_square && group_label[n1] ==  group_label[n2]) {
                NL[count++ * N + n1] = n2;
              }
            }
          }
        }
      }
    }
    NN[n1] = count;
  }
}

void find_neighbor_SW(
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const int* group_label,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;
  const double rc_cell_list = 0.5 * rc;
  const double rc_inv_cell_list = 2.0 / rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list, num_bins, box, position_per_atom, cell_count, cell_count_sum, cell_contents);

  gpu_find_neighbor_ON1_SW<<<grid_size, block_size>>>(
    box,
    N,
    N1,
    N2,
    type.data(),
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    NN.data(),
    NL.data(),
    group_label,
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv_cell_list,
    rc * rc);
  GPU_CHECK_KERNEL

  const int MN = NL.size() / NN.size();
  gpu_sort_neighbor_list<<<N, MN, MN * sizeof(int)>>>(N, NN.data(), NL.data());
  GPU_CHECK_KERNEL
}
