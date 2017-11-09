/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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



/*------------------------------------------------------------------------------
    This file will be idrectly included in some other files
------------------------------------------------------------------------------*/



/*------------------------------------------------------------------------------
    apply the minimum image convention
------------------------------------------------------------------------------*/


__device__ static void apply_mic
(
    int pbc_x, int pbc_y, int pbc_z, 
    real *b, // box matrix
    real *c, // inverse box matrix
    real &x12, real &y12, real &z12
) 
{
    real sx12 = c[0] * x12 + c[1] * y12 + c[2] * z12;
    real sy12 = c[3] * x12 + c[4] * y12 + c[5] * z12;
    real sz12 = c[6] * x12 + c[7] * y12 + c[8] * z12;
    if (pbc_x == 1) sx12 -= nearbyint(sx12);
    if (pbc_y == 1) sy12 -= nearbyint(sy12);
    if (pbc_z == 1) sz12 -= nearbyint(sz12);
    x12 = b[0] * sx12 + b[1] * sy12 + b[2] * sz12;
    y12 = b[3] * sx12 + b[4] * sy12 + b[5] * sz12;
    z12 = b[6] * sx12 + b[7] * sy12 + b[8] * sz12;
}


static __device__ void dev_apply_mic
(
    int pbc_x, int pbc_y, int pbc_z, real &x12, real &y12, real &z12, 
    real lx, real ly, real lz
)
{
    if      (pbc_x == 1 && x12 < - lx * HALF) {x12 += lx;}
    else if (pbc_x == 1 && x12 > + lx * HALF) {x12 -= lx;}
    if      (pbc_y == 1 && y12 < - ly * HALF) {y12 += ly;}
    else if (pbc_y == 1 && y12 > + ly * HALF) {y12 -= ly;}
    if      (pbc_z == 1 && z12 < - lz * HALF) {z12 += lz;}
    else if (pbc_z == 1 && z12 > + lz * HALF) {z12 -= lz;}
}



