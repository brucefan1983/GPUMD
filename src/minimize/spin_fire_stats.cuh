/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include <cmath>

namespace spin_fire_stats {
constexpr int BLOCK = 256;
// E, angular g^2/v^2/g.v, physical residual^2, invalid, relative update,
// then radial g^2/v^2/g.v (zero for fixed magnitudes).
constexpr int NUM_STATS = 10;

static __global__ void collect_stats(
  bool fixed,
  int n,
  const double* spin,
  const double* force,
  const double* energy,
  const double* gradient,
  const double* velocity,
  const double* previous_spin,
  double* result)
{
  __shared__ double sums[NUM_STATS][BLOCK];
  const int t = threadIdx.x;
  double e = 0, g2 = 0, v2 = 0, power = 0, gmax2 = 0, invalid = 0, change = 0;
  double radial_g2 = 0, radial_v2 = 0, radial_power = 0;
  for (int i = blockIdx.x * BLOCK + t; i < n; i += gridDim.x * BLOCK) {
    e += energy[i];
    if (!isfinite(energy[i]))
      invalid = 1;
    double unit[3] = {1, 0, 0}, gr = 0, vr = 0;
    if (!fixed) {
      double s2 = 0, gradient2 = 0;
      for (int c = 0; c < 3; ++c) {
        s2 += spin[c * n + i] * spin[c * n + i];
        gradient2 += gradient[c * n + i] * gradient[c * n + i];
      }
      const double length = sqrt(s2 > 0 ? s2 : gradient2);
      for (int c = 0; c < 3; ++c) {
        const int j = c * n + i;
        if (length > 0)
          unit[c] = (s2 > 0 ? spin[j] : gradient[j]) / length;
        gr += unit[c] * gradient[j];
        vr += unit[c] * velocity[j];
      }
      radial_g2 += gr * gr;
      radial_v2 += vr * vr;
      radial_power += gr * vr;
    }
    double gi2 = 0, physical2 = 0;
    for (int c = 0; c < 3; ++c) {
      const int j = c * n + i;
      const double g = gradient[j] - gr * unit[c], v = velocity[j] - vr * unit[c];
      if (!isfinite(spin[j]) || !isfinite(force[j]) || !isfinite(gradient[j]) || !isfinite(velocity[j]))
        invalid = 1;
      gi2 += g * g;
      physical2 += fixed ? gradient[j] * gradient[j] : force[j] * force[j];
      v2 += v * v;
      power += g * v;
      const double scale = fmax(1.0, fmax(fabs(spin[j]), fabs(previous_spin[j])));
      change = fmax(change, fabs(spin[j] - previous_spin[j]) / scale);
    }
    g2 += gi2;
    gmax2 = fmax(gmax2, physical2);
  }
  sums[0][t] = e;
  sums[1][t] = g2;
  sums[2][t] = v2;
  sums[3][t] = power;
  sums[4][t] = gmax2;
  sums[5][t] = invalid;
  sums[6][t] = change;
  sums[7][t] = radial_g2;
  sums[8][t] = radial_v2;
  sums[9][t] = radial_power;
  __syncthreads();
  for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
    if (t < offset) {
      for (int k = 0; k < NUM_STATS; ++k)
        sums[k][t] = (k >= 4 && k < 7) ? fmax(sums[k][t], sums[k][t + offset])
                                      : sums[k][t] + sums[k][t + offset];
    }
    __syncthreads();
  }
  if (t == 0)
    for (int k = 0; k < NUM_STATS; ++k)
      result[k * gridDim.x + blockIdx.x] = sums[k][0];
}


// Atomic force/velocity statistics and the full (possibly nonsymmetric) virial.
static __global__ void collect_lattice_stats(
  int n,
  const double* position,
  const double* previous,
  const double* force,
  const double* velocity,
  const double* virial,
  double* result)
{
  __shared__ double sums[15][BLOCK];
  const int t = threadIdx.x;
  double values[15] = {};
  for (int i = blockIdx.x * BLOCK + t; i < n; i += gridDim.x * BLOCK) {
    for (int c = 0; c < 3; ++c) {
      const int j = c * n + i;
      const double f = force[j], v = velocity[j];
      values[0] += f * f;
      values[1] += v * v;
      values[2] += f * v;
      values[3] = fmax(values[3], fabs(f));
      if (!isfinite(f) || !isfinite(v) || !isfinite(position[j]))
        values[4] = 1;
      const double scale = fmax(1.0, fmax(fabs(position[j]), fabs(previous[j])));
      values[5] = fmax(values[5], fabs(position[j] - previous[j]) / scale);
    }
    for (int c = 0; c < 9; ++c) {
      const double w = virial[c * n + i];
      if (!isfinite(w))
        values[4] = 1;
      values[6 + c] += w;
    }
  }
  for (int k = 0; k < 15; ++k)
    sums[k][t] = values[k];
  __syncthreads();
  for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
    if (t < offset)
      for (int k = 0; k < 15; ++k)
        sums[k][t] = (k >= 3 && k <= 5) ? fmax(sums[k][t], sums[k][t + offset])
                                        : sums[k][t] + sums[k][t + offset];
    __syncthreads();
  }
  if (t == 0)
    for (int k = 0; k < 15; ++k)
      result[k * gridDim.x + blockIdx.x] = sums[k][0];
}


// Partial statistics are field-major. Maxima/invalid flags retain their exact operation.
template<int Count, int FirstMax, int EndMax>
static __global__ void reduce_partials(int count, const double* partial, double* result) {
  __shared__ double sums[Count][BLOCK];
  const int t = threadIdx.x;
  for (int k = 0; k < Count; ++k) {
    double value = 0;
    for (int i = t; i < count; i += BLOCK)
      value = (k >= FirstMax && k < EndMax) ? fmax(value, partial[k * count + i])
                                           : value + partial[k * count + i];
    sums[k][t] = value;
  }
  __syncthreads();
  for (int offset = BLOCK / 2; offset > 0; offset /= 2) {
    if (t < offset)
      for (int k = 0; k < Count; ++k)
        sums[k][t] = (k >= FirstMax && k < EndMax) ? fmax(sums[k][t], sums[k][t + offset])
                                                : sums[k][t] + sums[k][t + offset];
    __syncthreads();
  }
  if (t == 0)
    for (int k = 0; k < Count; ++k) result[k] = sums[k][0];
}
} // namespace spin_fire_stats
