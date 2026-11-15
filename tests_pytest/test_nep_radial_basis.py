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

"""Bounds and value regression for the common NEP radial basis."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include <cassert>
#include <cmath>
#define __device__
#define __host__
#define __constant__
#define __forceinline__ inline
#include "utilities/nep_utilities.cuh"
int main() {
  for (int n : {0, 1, 8}) {
    float f[10], fp[10], only[10];
    for (int k=0; k<10; ++k) f[k]=fp[k]=only[k]=12345;
    find_fn(n, .2f, 1.7f, .8f, only);
    find_fn_and_fnp(n, .2f, 1.7f, .8f, -.1f, f, fp);
    assert(f[0]==.8f && fp[0]==-.1f);
    for (int k=0; k<=n; ++k) assert(std::abs(f[k]-only[k])<1e-6f);
    for (int k=n+1; k<10; ++k) assert(f[k]==12345 && fp[k]==12345 && only[k]==12345);
  }
}
'''


def test_radial_basis_bounds(tmp_path):
    source = tmp_path/'radial.cpp'
    binary = tmp_path/'radial'
    source.write_text('#include <initializer_list>\n'+SOURCE)
    subprocess.run(['c++', '-std=c++14', '-O0', '-I'+str(ROOT/'src'),
                    str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
