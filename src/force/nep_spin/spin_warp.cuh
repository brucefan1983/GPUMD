/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace nep_spin {
#ifdef USE_HIP
using WarpMask = unsigned long long;
#else
using WarpMask = unsigned int;
#endif

// A logical tile can occupy either half of an AMD wave64. Keep the mask in
// physical lane coordinates even when the algorithm groups atoms in tiles of 32.
__device__ __forceinline__ WarpMask tile_mask(int width) {
  const int first = (threadIdx.x % warpSize) / width * width;
  return (~WarpMask{0} >> (sizeof(WarpMask) * 8 - width)) << first;
}

__device__ __forceinline__ WarpMask warp_ballot(int predicate) {
  return __ballot_sync(~WarpMask{0}, predicate);
}
} // namespace nep_spin
