/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Major contributors:
  Chengbing Chen: Longitudinal spin wall formulation in TorchNEP.
  OpenAI Codex: GPUMD integration.
------------------------------------------------------------------------------*/

#pragma once
#include <cmath>

struct Spin_Wall {
  double threshold = 0.0;
  double stiffness = 0.0;
};

struct Spin_Walls {
  enum { high, low, count };
  Spin_Wall side[count];

  static const char* keyword(int i) { return i == high ? "spin_wall_high" : "spin_wall_low"; }
  bool enabled() const { return side[high].stiffness > 0 || side[low].stiffness > 0; }
  bool ordered() const
  {
    return side[high].stiffness == 0 || side[low].stiffness == 0 ||
           side[low].threshold < side[high].threshold;
  }
};

#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
  inline double
  evaluate_spin_wall(const Spin_Walls& walls, const double (&spin)[3], double (&mforce)[3])
{
  const double scale = fmax(fabs(spin[0]), fmax(fabs(spin[1]), fabs(spin[2])));
  const double divisor = scale > 0 ? scale : 1.0;
  const double x = spin[0] / divisor, y = spin[1] / divisor, z = spin[2] / divisor;
  const double root = sqrt(x * x + y * y + z * z);
  const double norm = scale * root;
  const auto hi = walls.side[Spin_Walls::high];
  const auto lo = walls.side[Spin_Walls::low];
  if (hi.stiffness > 0 && norm > hi.threshold) {
    const double excess = norm - hi.threshold;
    const double f = (hi.stiffness * excess) * excess;
    mforce[0] = -f * (x / root);
    mforce[1] = -f * (y / root);
    mforce[2] = -f * (z / root);
    return f * (excess / 3.0);
  }
  if (lo.stiffness > 0 && norm < lo.threshold) {
    const double a = spin[0] / lo.threshold, b = spin[1] / lo.threshold, c = spin[2] / lo.threshold;
    const double deficit = fmax(0.0, 1.0 - (a * a + b * b + c * c));
    const double f = lo.stiffness * deficit * deficit;
    mforce[0] = 2.0 * (f * a);
    mforce[1] = 2.0 * (f * b);
    mforce[2] = 2.0 * (f * c);
    return f * (lo.threshold / 3.0) * deficit;
  }
  mforce[0] = mforce[1] = mforce[2] = 0.0;
  return 0.0;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename Energy, typename MagneticForce>
static __global__ void add_spin_wall(
  int n,
  const int* type,
  const double* spin,
  const Spin_Walls* walls,
  Energy* energy,
  MagneticForce* mforce)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const auto& wall = walls[type[i]];
  if (wall.side[Spin_Walls::high].stiffness == 0 && wall.side[Spin_Walls::low].stiffness == 0)
    return;
  const double moment[3] = {spin[i], spin[n + i], spin[2 * n + i]};
  double field[3];
  energy[i] += evaluate_spin_wall(wall, moment, field);
  for (int d = 0; d < 3; ++d)
    mforce[d * n + i] += field[d];
}
#endif
