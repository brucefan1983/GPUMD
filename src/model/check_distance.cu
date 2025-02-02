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
#include <cmath>

void applyMicOne(double& x12)
{
  if (x12 < -0.5) {
    x12 += 1.0;
  } else if (x12 > +0.5) {
    x12 -= 1.0;
  }
}

void applyMic(const Box& box, double& x12, double& y12, double& z12)
{
  int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  double s[3];
  for (int i = 0; i < 3; ++i) {
    s[i] = box.cpu_h[9 + i * 3] * x12 + box.cpu_h[10 + i * 3] * y12 + box.cpu_h[11 + i * 3] * z12;
    if (pbc[i])
      applyMicOne(s[i]);
  }
  x12 = box.cpu_h[0] * s[0] + box.cpu_h[3] * s[1] + box.cpu_h[6] * s[2];
  y12 = box.cpu_h[1] * s[0] + box.cpu_h[4] * s[1] + box.cpu_h[7] * s[2];
  z12 = box.cpu_h[2] * s[0] + box.cpu_h[5] * s[1] + box.cpu_h[8] * s[2];
}

void findCell(
  const Box& box, const double* thickness, const double* r, const int* numCells, int* cell)
{
  double s[3];
  for (int d = 0; d < 3; ++d) {
    s[d] =
      box.cpu_h[9 + d * 3] * r[0] + box.cpu_h[10 + d * 3] * r[1] + box.cpu_h[11 + d * 3] * r[2];
    cell[d] = floor(s[d] * thickness[d] * 0.2);
    if (cell[d] < 0)
      cell[d] += numCells[d];
    if (cell[d] >= numCells[d])
      cell[d] -= numCells[d];
  }
  cell[3] = cell[0] + numCells[0] * (cell[1] + numCells[1] * cell[2]);
}

void calculate_min_atomic_distance(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const double* pos = atom.cpu_position_per_atom.data();

  double min_distance = 5.0;
  int min_n1 = -1, min_n2 = -1;

  int cell[4], numCells[4];
  double thickness[3];
  for (int i = 0; i < 3; ++i) {
    thickness[i] = sqrt(
      box.cpu_h[i] * box.cpu_h[i] + box.cpu_h[i + 3] * box.cpu_h[i + 3] +
      box.cpu_h[i + 6] * box.cpu_h[i + 6]);
    numCells[i] = std::max(1, static_cast<int>(ceil(thickness[i] * 0.2)));
  }
  numCells[3] = numCells[0] * numCells[1] * numCells[2];

  std::vector<int> cellContents(N, 0);
  std::vector<int> cellCount(numCells[3], 0);
  std::vector<int> cellCountSum(numCells[3], 0);
  std::fill(cellCount.begin(), cellCount.end(), 0);

  for (int n = 0; n < N; ++n) {
    const double r[3] = {pos[n], pos[n + N], pos[n + 2 * N]};
    findCell(box, thickness, r, numCells, cell);
    ++cellCount[cell[3]];
  }

  for (int i = 1; i < numCells[3]; ++i) {
    cellCountSum[i] = cellCountSum[i - 1] + cellCount[i - 1];
  }

  for (int n = 0; n < N; ++n) {
    const double r[3] = {pos[n], pos[n + N], pos[n + 2 * N]};
    findCell(box, thickness, r, numCells, cell);
    cellContents[cellCountSum[cell[3]] + cellCount[cell[3]]] = n;
    ++cellCount[cell[3]];
  }

  for (int n1 = 0; n1 < N; ++n1) {
    const double r1[3] = {pos[n1], pos[n1 + N], pos[n1 + 2 * N]};
    findCell(box, thickness, r1, numCells, cell);
    for (int k = -1; k <= 1; ++k) {
      for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
          int neighborCell = cell[3] + (k * numCells[1] + j) * numCells[0] + i;
          if (cell[0] + i < 0)
            neighborCell += numCells[0];
          if (cell[0] + i >= numCells[0])
            neighborCell -= numCells[0];
          if (cell[1] + j < 0)
            neighborCell += numCells[1] * numCells[0];
          if (cell[1] + j >= numCells[1])
            neighborCell -= numCells[1] * numCells[0];
          if (cell[2] + k < 0)
            neighborCell += numCells[3];
          if (cell[2] + k >= numCells[2])
            neighborCell -= numCells[3];
          for (int m = 0; m < cellCount[neighborCell]; ++m) {
            const int n2 = cellContents[cellCountSum[neighborCell] + m];
            if (n1 < n2) {
              double x12 = pos[n2] - r1[0];
              double y12 = pos[n2 + N] - r1[1];
              double z12 = pos[n2 + 2 * N] - r1[2];
              applyMic(box, x12, y12, z12);
              if (fabs(x12) > 2.0 || fabs(y12) > 2.0 || fabs(z12) > 2.0)
                continue;
              const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
              if (d2 >= 4.0)
                continue;

              double distance = d2;
              if (distance < min_distance) {
                min_distance = distance;
                min_n1 = n1;
                min_n2 = n2;
              }
            }
          }
        }
      }
    }
  }
  double mini_distance = sqrt(min_distance);

  if (mini_distance < 1.0) {
    printf(
      "Error: Minimum distance (%f Å) between atoms %d (%s) and %d (%s) is less than 1 Å.\n",
      mini_distance,
      min_n1,
      atom.cpu_atom_symbol[min_n1].c_str(),
      min_n2,
      atom.cpu_atom_symbol[min_n2].c_str());
    PRINT_INPUT_ERROR("There are two atoms with a distance less than 1 Å.");
  } else if (min_n1 != -1 && min_n2 != -1) {
    printf(
      "Minimum distance between atoms %d (%s) and %d (%s): %f Å\n",
      min_n1,
      atom.cpu_atom_symbol[min_n1].c_str(),
      min_n2,
      atom.cpu_atom_symbol[min_n2].c_str(),
      mini_distance);
  }
}
