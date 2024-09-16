/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
mininum image convention
------------------------------------------------------------------------------*/

static __device__ void
dev_apply_mic(const float* __restrict__ h, float& x12, float& y12, float& z12)
{
  float sx12 = h[9] * x12 + h[10] * y12 + h[11] * z12;
  float sy12 = h[12] * x12 + h[13] * y12 + h[14] * z12;
  float sz12 = h[15] * x12 + h[16] * y12 + h[17] * z12;
  sx12 -= nearbyint(sx12);
  sy12 -= nearbyint(sy12);
  sz12 -= nearbyint(sz12);
  x12 = h[0] * sx12 + h[1] * sy12 + h[2] * sz12;
  y12 = h[3] * sx12 + h[4] * sy12 + h[5] * sz12;
  z12 = h[6] * sx12 + h[7] * sy12 + h[8] * sz12;
}
