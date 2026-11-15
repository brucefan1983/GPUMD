# Copyright 2017 Zheyong Fan and GPUMD development team
# This file is part of GPUMD.
# GPUMD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# GPUMD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.

"""CPU mask contract checks; GPU shuffle semantics use validate_spin_warp.cu."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('hip,wave', [(False, 32), (True, 32), (True, 64)])
def test_physical_tile_masks_and_ballot(tmp_path, hip, wave):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler unavailable')
    # Only the device environment is stubbed. The production mask helper is
    # compiled unchanged; this does not claim to emulate GPU synchronization.
    runtime = tmp_path / ('hip/hip_runtime.h' if hip else 'cuda_runtime.h')
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text('''#pragma once
#define __device__
#define __forceinline__ inline
struct { unsigned int x; } threadIdx;
int warpSize;
#ifdef USE_HIP
using expected_mask = unsigned long long;
#else
using expected_mask = unsigned;
#endif
expected_mask active_lanes;
template<class Mask> expected_mask __ballot_sync(Mask mask, int predicate) {
  static_assert(sizeof(Mask) == sizeof(expected_mask), "wrong platform mask width");
  return predicate ? active_lanes & mask : 0;
}
''')
    main = tmp_path / 'main.cpp'
    main.write_text('''#include "force/nep_spin/spin_warp.cuh"
#include <cassert>
int main() {
  warpSize = WAVE;
  static_assert(sizeof(nep_spin::WarpMask) == sizeof(expected_mask), "ballot truncation");
  for (int width : {1, 4, 8, 32}) {
    for (unsigned id = 0; id < 256; ++id) {
      threadIdx.x = id;
      unsigned long long expected = 0;
      for (unsigned other = 0; other < 256; ++other)
        if (other / width == id / width)
          expected |= 1ull << (other % warpSize);
      assert(nep_spin::tile_mask(width) == expected);
    }
  }
  active_lanes = static_cast<expected_mask>(0x80000101000000a5ull);
  assert(nep_spin::warp_ballot(1) == active_lanes);
  assert(nep_spin::warp_ballot(0) == 0);
}
'''.replace('#include <cassert>', '#include <cassert>\n#include <initializer_list>'))
    command = [compiler, '-std=c++14', '-pedantic-errors', f'-DWAVE={wave}',
               '-I' + str(tmp_path), '-I' + str(ROOT / 'src'), str(main), '-o', str(tmp_path / 'check')]
    if hip:
        command.insert(1, '-DUSE_HIP')
    subprocess.run(command, check=True)
    subprocess.run([str(tmp_path / 'check')], check=True)
