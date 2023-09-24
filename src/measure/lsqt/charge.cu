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

#include "charge.cuh"
#include <fstream>
#include <iostream>
#include <limits.h>

void Charge::add_impurities(
  std::mt19937& generator,
  int number_of_atoms,
  real box_length[3],
  int pbc[3],
  std::vector<real>& x,
  std::vector<real>& y,
  std::vector<real>& z,
  real* potential)
{
  rc = 5.0 * xi; // exp(-25/2) is of the order of 1.0e-6
  rc2 = rc * rc;
  impurity_indices.resize(Ni);
  impurity_strength.resize(Ni);
  find_impurity_indices(generator, number_of_atoms);
  find_impurity_strength(generator);
  find_cell_numbers(pbc, box_length);
  find_cell_contents(number_of_atoms, pbc, box_length, x, y, z);
  find_potentials(number_of_atoms, box_length, pbc, x, y, z, potential);
}

void Charge::find_impurity_indices(std::mt19937& generator, int max_value)
{
  int* permuted_numbers = new int[max_value];
  for (int i = 0; i < max_value; ++i) {
    permuted_numbers[i] = i;
  }
  std::uniform_int_distribution<int> rand_int(0, INT_MAX);
  for (int i = 0; i < max_value; ++i) {
    int j = rand_int(generator) % (max_value - i) + i;
    int temp = permuted_numbers[i];
    permuted_numbers[i] = permuted_numbers[j];
    permuted_numbers[j] = temp;
  }
  for (int i = 0; i < Ni; ++i) {
    impurity_indices[i] = permuted_numbers[i];
  }
  delete[] permuted_numbers;
}

void Charge::find_impurity_strength(std::mt19937& generator)
{
  real W2 = W * 0.5;
  std::uniform_real_distribution<real> strength(-W2, W2);
  for (int i = 0; i < Ni; ++i) {
    impurity_strength[i] = strength(generator);
  }
}

void Charge::find_cell_numbers(int pbc[3], real box_length[3])
{
  if (pbc[0])
    Nx = floor(box_length[0] / rc);
  else
    Nx = 1;
  if (pbc[1])
    Ny = floor(box_length[1] / rc);
  else
    Ny = 1;
  if (pbc[2])
    Nz = floor(box_length[2] / rc);
  else
    Nz = 1;
  Nxyz = Nx * Ny * Nz;
}

void Charge::find_cell_contents(
  int N,
  int pbc[3],
  real box_length[3],
  std::vector<real>& x,
  std::vector<real>& y,
  std::vector<real>& z)
{
  cell_count.assign(Nxyz, 0);
  cell_count_sum.assign(Nxyz, 0);
  for (int i = 0; i < N; ++i) {
    int nxyz = find_cell_id(x[i], y[i], z[i], rc);
    ++cell_count[nxyz];
  }
  for (int i = 1; i < Nxyz; ++i) {
    cell_count_sum[i] = cell_count[i - 1] + cell_count_sum[i - 1];
  }
  cell_count.assign(Nxyz, 0);
  cell_contents.assign(N, 0);
  for (int i = 0; i < N; ++i) {
    int nxyz = find_cell_id(x[i], y[i], z[i], rc);
    cell_contents[cell_count_sum[nxyz] + cell_count[nxyz]] = i;
    ++cell_count[nxyz];
  }
}

static real find_d12_square(int pbc[3], real box_length[3], real box_length_half[3], real r12[3])
{
  real d12_square = 0.0;
  for (int d = 0; d < 3; ++d) {
    r12[d] = fabs(r12[d]);
    if (pbc[d] == 1 && r12[d] > box_length_half[d]) {
      r12[d] = box_length[d] - r12[d];
    }
    d12_square += r12[d] * r12[d];
  }
  return d12_square;
}

void Charge::find_potentials(
  int number_of_atoms,
  real box_length[3],
  int pbc[3],
  std::vector<real>& x,
  std::vector<real>& y,
  std::vector<real>& z,
  real* potential)
{
  real xi_factor = -0.5 / (xi * xi);
  real box_length_half[3];
  for (int d = 0; d < 3; ++d)
    box_length_half[d] = box_length[d] * 0.5;
  for (int n = 0; n < number_of_atoms; ++n)
    potential[n] = 0.0;
  for (int ni = 0; ni < Ni; ++ni) {
    int n1 = impurity_indices[ni];
    real x1 = x[n1];
    real y1 = y[n1];
    real z1 = z[n1];
    int nx, ny, nz, nxyz;
    find_cell_id(x1, y1, z1, rc, nx, ny, nz, nxyz);
    int klim = pbc[2] ? 1 : 0;
    int jlim = pbc[1] ? 1 : 0;
    int ilim = pbc[0] ? 1 : 0;
    for (int k = -klim; k < klim + 1; ++k)
      for (int j = -jlim; j < jlim + 1; ++j)
        for (int i = -ilim; i < ilim + 1; ++i) {
          int neighbor = find_neighbor_cell(nx, ny, nz, nxyz, i, j, k);
          for (int m = 0; m < cell_count[neighbor]; ++m) {
            int n2 = cell_contents[cell_count_sum[neighbor] + m];
            real r12[3];
            r12[0] = x[n2] - x1;
            r12[1] = y[n2] - y1;
            r12[2] = z[n2] - z1;
            real d12_square = find_d12_square(pbc, box_length, box_length_half, r12);
            if (d12_square > rc2)
              continue;
            potential[n2] += impurity_strength[ni] * exp(d12_square * xi_factor);
          }
        }
  }
}

int Charge::find_cell_id(real x, real y, real z, real rc)
{
  int nx = floor(x / rc);
  int ny = floor(y / rc);
  int nz = floor(z / rc);
  while (nx < 0)
    nx += Nx;
  while (nx >= Nx)
    nx -= Nx;
  while (ny < 0)
    ny += Ny;
  while (ny >= Ny)
    ny -= Ny;
  while (nz < 0)
    nz += Nz;
  while (nz >= Nz)
    nz -= Nz;
  int nxyz = nx + Nx * ny + Nx * Ny * nz;
  return nxyz;
}

void Charge::find_cell_id(real x, real y, real z, real rc, int& nx, int& ny, int& nz, int& nxyz)
{
  nx = floor(x / rc);
  ny = floor(y / rc);
  nz = floor(z / rc);
  while (nx < 0)
    nx += Nx;
  while (nx >= Nx)
    nx -= Nx;
  while (ny < 0)
    ny += Ny;
  while (ny >= Ny)
    ny -= Ny;
  while (nz < 0)
    nz += Nz;
  while (nz >= Nz)
    nz -= Nz;
  nxyz = nx + Nx * ny + Nx * Ny * nz;
}

int Charge::find_neighbor_cell(int nx, int ny, int nz, int nxyz, int i, int j, int k)
{
  int neighbor = nxyz + k * Nx * Ny + j * Nx + i;
  if (nx + i < 0)
    neighbor += Nx;
  if (nx + i >= Nx)
    neighbor -= Nx;
  if (ny + j < 0)
    neighbor += Ny * Nx;
  if (ny + j >= Ny)
    neighbor -= Ny * Nx;
  if (nz + k < 0)
    neighbor += Nxyz;
  if (nz + k >= Nz)
    neighbor -= Nxyz;
  return neighbor;
}
