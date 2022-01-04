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
The class defining the simulation box.
------------------------------------------------------------------------------*/

#include "box.cuh"
#include "utilities/error.cuh"
#include <cmath>

double Box::get_volume(void)
{
  double volume;
  if (triclinic) {
    volume = abs(
      cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
      cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
      cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]));
  } else {
    volume = cpu_h[0] * cpu_h[1] * cpu_h[2];
  }
  return volume;
}

void Box::get_inverse(void)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  double det;
  if (triclinic) {
    det = cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
          cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
          cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
  } else {
    det = cpu_h[0] * cpu_h[1] * cpu_h[2];
  }
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= det;
  }
}

void static get_num_bins_one_direction(
  const int pbc, const double rc, const double box_length, int& num_bins, bool& use_ON2)
{
  if (pbc) {
    num_bins = floor(box_length / rc);
    if (num_bins < 3) {
      use_ON2 = true;
    }
  } else {
    num_bins = 1;
  }
}

bool Box::get_num_bins(const double rc, int num_bins[]) const
{
  bool use_ON2 = false;
  if (triclinic) {
    use_ON2 = true;
  } else {
    get_num_bins_one_direction(pbc_x, rc, cpu_h[0], num_bins[0], use_ON2);
    get_num_bins_one_direction(pbc_y, rc, cpu_h[1], num_bins[1], use_ON2);
    get_num_bins_one_direction(pbc_z, rc, cpu_h[2], num_bins[2], use_ON2);
  }
  if (num_bins[0] * num_bins[1] * num_bins[2] < 50) {
    use_ON2 = true;
  }
  return use_ON2;
}
