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

#pragma once

#include "model/box.cuh"

static __device__ void get_fractional_position(
  const Box& box,
  const double x,
  const double y,
  const double z,
  double& sa,
  double& sb,
  double& sc)
{
  sa = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  sb = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  sc = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
}

static __device__ bool is_in_region(
  const double sa,
  const double sb,
  const double sc,
  const double amin,
  const double amax,
  const double bmin,
  const double bmax,
  const double cmin,
  const double cmax)
{
  return sa >= amin && sa < amax && sb >= bmin && sb < bmax && sc >= cmin && sc < cmax;
}
