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
Calculate the distance between any two atoms in the model.xyz file.
------------------------------------------------------------------------------*/

#include "atom.cuh"
#include "box.cuh"
#include "check_distance.cuh"
#include "utilities/error.cuh"
#include <array>
#include <cmath>
#include <vector>

void calculate_min_atomic_distance(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const double* pos = atom.cpu_position_per_atom.data();

  double min_distance = 5.0;
  int min_i = -1, min_j = -1;

  double Lx =
    sqrt(box.cpu_h[0] * box.cpu_h[0] + box.cpu_h[1] * box.cpu_h[1] + box.cpu_h[2] * box.cpu_h[2]);
  double Ly =
    sqrt(box.cpu_h[3] * box.cpu_h[3] + box.cpu_h[4] * box.cpu_h[4] + box.cpu_h[5] * box.cpu_h[5]);
  double Lz =
    sqrt(box.cpu_h[6] * box.cpu_h[6] + box.cpu_h[7] * box.cpu_h[7] + box.cpu_h[8] * box.cpu_h[8]);

  int nx = std::max(1, static_cast<int>(ceil(Lx * 0.2)));
  int ny = std::max(1, static_cast<int>(ceil(Ly * 0.2)));
  int nz = std::max(1, static_cast<int>(ceil(Lz * 0.2)));
  const double dx = Lx / nx, dy = Ly / ny, dz = Lz / nz;

  std::vector<std::array<double, 3>> wrapped_pos(N);
  for (int i = 0; i < N; ++i) {
    double x = pos[i];
    double y = pos[i + N];
    double z = pos[i + 2 * N];

    double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;

    if (box.pbc_x)
      sx -= floor(sx);
    if (box.pbc_y)
      sy -= floor(sy);
    if (box.pbc_z)
      sz -= floor(sz);

    wrapped_pos[i][0] = sx * box.cpu_h[0] + sy * box.cpu_h[3] + sz * box.cpu_h[6];
    wrapped_pos[i][1] = sx * box.cpu_h[1] + sy * box.cpu_h[4] + sz * box.cpu_h[7];
    wrapped_pos[i][2] = sx * box.cpu_h[2] + sy * box.cpu_h[5] + sz * box.cpu_h[8];
  }

  std::vector<std::vector<int>> grid(nx * ny * nz);

  auto get_cell_index = [&](const std::array<double, 3>& p) {
    int ix = static_cast<int>((p[0] / Lx) * nx) % nx;
    int iy = static_cast<int>((p[1] / Ly) * ny) % ny;
    int iz = static_cast<int>((p[2] / Lz) * nz) % nz;
    return ix + iy * nx + iz * nx * ny;
  };

  for (int i = 0; i < N; ++i) {
    int cell = get_cell_index(wrapped_pos[i]);
    grid[cell].push_back(i);
  }

  for (int i = 0; i < N; ++i) {
    const auto& pi = wrapped_pos[i];
    int cx = static_cast<int>((pi[0] / Lx) * nx) % nx;
    int cy = static_cast<int>((pi[1] / Ly) * ny) % ny;
    int cz = static_cast<int>((pi[2] / Lz) * nz) % nz;

    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          int nx_cell = (cx + dx + nx) % nx;
          int ny_cell = (cy + dy + ny) % ny;
          int nz_cell = (cz + dz + nz) % nz;

          if (!box.pbc_x && (cx + dx < 0 || cx + dx >= nx))
            continue;
          if (!box.pbc_y && (cy + dy < 0 || cy + dy >= ny))
            continue;
          if (!box.pbc_z && (cz + dz < 0 || cz + dz >= nz))
            continue;

          int neighbor_cell = nx_cell + ny_cell * nx + nz_cell * nx * ny;

          for (int j : grid[neighbor_cell]) {
            if (j <= i)
              continue;

            const auto& pj = wrapped_pos[j];
            double delta[3] = {pi[0] - pj[0], pi[1] - pj[1], pi[2] - pj[2]};

            if (fabs(delta[0]) > 2.0 || fabs(delta[1]) > 2.0 || fabs(delta[2]) > 2.0)
              continue;

            if (box.pbc_x) {
              delta[0] -= box.cpu_h[0] * std::round(
                                           box.cpu_h[9] * delta[0] + box.cpu_h[10] * delta[1] +
                                           box.cpu_h[11] * delta[2]);
            }
            if (box.pbc_y) {
              delta[1] -= box.cpu_h[3] * std::round(
                                           box.cpu_h[12] * delta[0] + box.cpu_h[13] * delta[1] +
                                           box.cpu_h[14] * delta[2]);
            }
            if (box.pbc_z) {
              delta[2] -= box.cpu_h[6] * std::round(
                                           box.cpu_h[15] * delta[0] + box.cpu_h[16] * delta[1] +
                                           box.cpu_h[17] * delta[2]);
            }

            double dist_sq = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
            if (dist_sq >= 4.0)
              continue;

            double dist = sqrt(dist_sq);
            if (dist < min_distance) {
              min_distance = dist;
              min_i = i;
              min_j = j;
            }
          }
        }
      }
    }
  }

  if (min_distance < 1.0) {
    printf(
      "Error: Minimum distance (%f Å) between atoms %d (%s) and %d (%s) is less than 1 Å.\n",
      min_distance,
      min_i,
      atom.cpu_atom_symbol[min_i].c_str(),
      min_j,
      atom.cpu_atom_symbol[min_j].c_str());
    PRINT_INPUT_ERROR("There are two atoms with a distance less than 1 Å.");
  } else if (min_i != -1 && min_j != -1) {
    printf(
      "Minimum distance between atoms %d (%s) and %d (%s): %f Å\n",
      min_i,
      atom.cpu_atom_symbol[min_i].c_str(),
      min_j,
      atom.cpu_atom_symbol[min_j].c_str(),
      min_distance);
  }
}
