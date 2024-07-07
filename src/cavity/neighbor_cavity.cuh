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

#pragma once
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"

void find_cell_list_jacobian(
  const double rc,
  const int* num_bins,
  const int num_copies,
  Box& box,
  const GPU_Vector<double>& position_per_atom,
  const GPU_Vector<int>& system_index,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents);

static __device__ void find_cell_id_jacobian(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  const int system_index,
  const int number_of_cells,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  if (box.triclinic == 0) {
    cell_id_x = floor(x * rc_inv);
    cell_id_y = floor(y * rc_inv);
    cell_id_z = floor(z * rc_inv);
  } else {
    const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    cell_id_x = floor(sx * box.thickness_x * rc_inv);
    cell_id_y = floor(sy * box.thickness_y * rc_inv);
    cell_id_z = floor(sz * box.thickness_z * rc_inv);
  }
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
  // Additionally offset the cell id by system_index * number_of_cells
  // such that only atoms belonging to the same copy of the system
  // are in neighboring cells. 
  cell_id += system_index * number_of_cells;
}
