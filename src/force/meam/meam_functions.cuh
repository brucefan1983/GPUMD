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
/*
    Contributors
    OpenAI Codex: Initial implementation
*/

#pragma once

#include <cmath>

struct MeamValueSlope {
  double value;
  double slope;
};

// Smooth MEAM cutoff, with the derivative taken with respect to its unit-interval argument.
static __host__ __device__ inline MeamValueSlope meam_cutoff(double x)
{
  if (x <= 0.0) {
    return {0.0, 0.0};
  }
  if (x >= 1.0) {
    return {1.0, 0.0};
  }
  const double y = 1.0 - x;
  const double y2 = y * y;
  const double f = 1.0 - y2 * y2;
  return {f * f, 8.0 * y2 * y * f};
}

static __host__ __device__ inline MeamValueSlope
meam_angular_scaling(double gamma, int form, double smoothing)
{
  if (form == 3) {
    const double g = 2.0 / (1.0 + exp(-gamma));
    return {g, g * (1.0 - 0.5 * g)};
  }
  if (form == 1) {
    const double g = exp(0.5 * gamma);
    return {g, 0.5 * g};
  }
  if (form == -5) {
    const double x = 1.0 + gamma;
    const double g = sqrt(fabs(x));
    return {copysign(g, x), g > 0.0 ? 0.5 / g : 0.0};
  }
  const double limit = -smoothing / (smoothing + 1.0);
  if (gamma < limit) {
    const double g = sqrt(pow(limit / gamma, smoothing) / (smoothing + 1.0));
    return {g, -0.5 * smoothing * g / gamma};
  }
  const double g = sqrt(1.0 + gamma);
  return {g, 0.5 / g};
}

static __host__ __device__ inline MeamValueSlope
meam_embedding(double density, double scale, bool linear_negative)
{
  if (density > 0.0) {
    const double logarithm = log(density);
    return {scale * density * logarithm, scale * (1.0 + logarithm)};
  }
  return linear_negative ? MeamValueSlope{-scale * density, -scale} : MeamValueSlope{0.0, 0.0};
}
