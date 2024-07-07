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

/*----------------------------------------------------------------------------80
neighbor list.
------------------------------------------------------------------------------*/

#include "cavity/neighbor_cavity.cuh"
#include "utilities/error.cuh"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

static __device__ void find_cell_id_jacobian(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int system_index,
  const int number_of_cells,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id_jacobian(box, 
      x, 
      y, 
      z, 
      rc_inv, 
      nx, 
      ny, 
      nz, 
      system_index, 
      number_of_cells,
      cell_id_x, 
      cell_id_y, 
      cell_id_z, 
      cell_id);
}

static __global__ void find_cell_counts_jacobian(
  const Box box,
  const int N,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int* system_index,
  const int number_of_cells,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id_jacobian(box, x[n1], y[n1], z[n1], rc_inv, system_index[n1], number_of_cells, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents_jacobian(
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int* system_index,
  const int number_of_cells,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id_jacobian(box, x[n1], y[n1], z[n1], rc_inv, system_index[n1], number_of_cells, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}


void find_cell_list_jacobian(
  const double rc,
  const int* num_bins,
  const int num_copies,
  Box& box,
  const GPU_Vector<double>& position_per_atom,
  const GPU_Vector<int>& system_index,
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
  const int N_cells_per_copy = num_bins[0] * num_bins[1] * num_bins[2];
  const int N_cells = N_cells_per_copy * num_copies;


  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  CHECK(cudaMemset(cell_count.data(), 0, sizeof(int) * N_cells));
  CHECK(cudaMemset(cell_count_sum.data(), 0, sizeof(int) * N_cells));
  CHECK(cudaMemset(cell_contents.data(), 0, sizeof(int) * N));

find_cell_counts_jacobian<<<grid_size, block_size>>>(
    box, N, cell_count.data(), x, y, z, system_index.data(), N_cells_per_copy, num_bins[0], num_bins[1], num_bins[2], rc_inv);
  CUDA_CHECK_KERNEL

  thrust::exclusive_scan(
    thrust::device, cell_count.data(), cell_count.data() + N_cells, cell_count_sum.data());

  CHECK(cudaMemset(cell_count.data(), 0, sizeof(int) * N_cells));

  find_cell_contents_jacobian<<<grid_size, block_size>>>(
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    system_index.data(), 
    N_cells_per_copy,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  CUDA_CHECK_KERNEL
}
