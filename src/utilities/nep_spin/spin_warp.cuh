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
  Chengbing Chen: Initial implementation.
------------------------------------------------------------------------------*/

#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace nep_spin
{
#ifdef USE_HIP
using WarpMask = unsigned long long;
#else
using WarpMask = unsigned int;
#endif

// A logical tile can occupy either half of an AMD wave64. Keep the mask in
// physical lane coordinates even when the algorithm groups atoms in tiles of 32.
__device__ __forceinline__ WarpMask tile_mask(int width)
{
  const int first = (threadIdx.x % warpSize) / width * width;
  return (~WarpMask{0} >> (sizeof(WarpMask) * 8 - width)) << first;
}

__device__ __forceinline__ WarpMask warp_ballot(int predicate)
{
  return __ballot_sync(~WarpMask{0}, predicate);
}
} // namespace nep_spin
