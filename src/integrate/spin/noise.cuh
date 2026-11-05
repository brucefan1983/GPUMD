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
Spin noise implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com
*/

#pragma once
#include "utilities/common.cuh"
#include <cmath>
#include <cstdint>

namespace
{
__device__ inline std::uint64_t splitmix64(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ inline double uniform_01(std::uint64_t& state)
{
  state = splitmix64(state);
  return (state >> 11) * (1.0 / 9007199254740992.0);
}

__device__ inline double gaussian(
  const std::uint64_t seed,
  const std::uint64_t atom_index,
  const std::uint64_t step,
  const int component)
{
  std::uint64_t state = seed;
  state ^= atom_index * 0xd1b54a32d192ed03ULL;
  state ^= step * 0x9e3779b97f4a7c15ULL;
  state ^= static_cast<std::uint64_t>(component) * 0x94d049bb133111ebULL;
  double u1 = 0.0;
  do {
    u1 = uniform_01(state);
  } while (u1 <= 0.0);
  const double u2 = uniform_01(state);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

} // namespace
