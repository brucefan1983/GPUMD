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
#include "utilities/gpu_vector.cuh"

class Potential
{
public:
  int N1;
  int N2;
  double rc; // maximum cutoff distance
  Potential(void);
  virtual ~Potential(void);

  virtual void compute(
    const int type_shift,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial) = 0;

protected:
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;

  void find_properties_many_body(
    Box& box,
    const int* NN,
    const int* NL,
    const double* f12x,
    const double* f12y,
    const double* f12z,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void find_cell_list(const int* num_bins, Box& box, const GPU_Vector<double>& position);

  void find_neighbor(
    Box& box,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& NN,
    GPU_Vector<int>& NL);
};

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
}
