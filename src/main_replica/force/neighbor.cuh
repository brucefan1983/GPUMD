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

#pragma once
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>

class GPU_Exclusive_Scan;

void find_cell_list(
  const double rc,
  const int* num_bins,
  Box& box,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents);

void find_cell_list(
  const gpuStream_t stream,
  const double rc,
  const int* num_bins,
  Box& box,
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Exclusive_Scan& scan_workspace);

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
  GPU_Vector<int>& NL);

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
  GPU_Vector<int>& NL,
  GPU_Exclusive_Scan& scan_workspace,
  const gpuStream_t stream);

// For ILP
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
  GPU_Vector<int>& big_ilp_NL);

// for SW
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
  GPU_Vector<int>& NL);

static __device__ void find_cell_id(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  cell_id_x = floor(sx * box.thickness_x * rc_inv);
  cell_id_y = floor(sy * box.thickness_y * rc_inv);
  cell_id_z = floor(sz * box.thickness_z * rc_inv);

  while (cell_id_x < 0)
    cell_id_x += nx;
  while (cell_id_x >= nx)
    cell_id_x -= nx;
  while (cell_id_y < 0)
    cell_id_y += ny;
  while (cell_id_y >= ny)
    cell_id_y -= ny;
  while (cell_id_z < 0)
    cell_id_z += nz;
  while (cell_id_z >= nz)
    cell_id_z -= nz;
  cell_id = cell_id_x + nx * cell_id_y + nx * ny * cell_id_z;
}

static __global__ void gpu_sort_neighbor_list(const int N, const int* NN, int* NL)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int neighbor_number = NN[bid];
  extern __shared__ int atom_index_copy[];

  for (int i = tid; i < neighbor_number; i += blockDim.x) {
    atom_index_copy[i] = NL[static_cast<size_t>(N) * i + bid];
  }
  __syncthreads();

  for (int i = tid; i < neighbor_number; i += blockDim.x) {
    int atom_index = atom_index_copy[i];
    int count = 0;
    for (int j = 0; j < neighbor_number; ++j) {
      if (atom_index > atom_index_copy[j]) {
        count++;
      }
    }
    NL[static_cast<size_t>(N) * count + bid] = atom_index;
  }
}

class Neighbor
{
public:
  Neighbor();
  ~Neighbor();
  Neighbor(const Neighbor&) = delete;
  Neighbor& operator=(const Neighbor&) = delete;

  GPU_Vector<int> NN, NL; // global neighbor list
  void initialize(const double rc, const int num_atoms, const int num_neighbors);
  void find_neighbor_global(
    const double rc,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom);
  void find_neighbor_global(
    const double rc,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    const gpuStream_t stream);
  void find_local_neighbor_from_global(
    const double rc,
    Box& box,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& NN_local,
    GPU_Vector<int>& NL_local);

private:
  double skin = 1.0;              // skin distance
  GPU_Vector<int> cell_count;     // for cell list
  GPU_Vector<int> cell_count_sum; // for cell list
  GPU_Vector<int> cell_contents;  // for cell list
  std::unique_ptr<GPU_Exclusive_Scan> scan_workspace;
  GPU_Vector<double> x0, y0, z0;  // for checking atom distance
  GPU_Vector<int> atom_distance_flag;
  int check_atom_distance(Box& box, const double* x, const double* y, const double* z);
};
