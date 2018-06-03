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
    This file will be directly included in some other files
------------------------------------------------------------------------------*/



/*------------------------------------------------------------------------------
    apply the minimum image convention
------------------------------------------------------------------------------*/
template <int pbc_x, int pbc_y, int pbc_z>
static __device__ void dev_apply_mic
(real lx, real ly, real lz, real *x12, real *y12, real *z12)
{
    if      (pbc_x == 1 && *x12 < - lx * HALF) {*x12 += lx;}
    else if (pbc_x == 1 && *x12 > + lx * HALF) {*x12 -= lx;}
    if      (pbc_y == 1 && *y12 < - ly * HALF) {*y12 += ly;}
    else if (pbc_y == 1 && *y12 > + ly * HALF) {*y12 -= ly;}
    if      (pbc_z == 1 && *z12 < - lz * HALF) {*z12 += lz;}
    else if (pbc_z == 1 && *z12 > + lz * HALF) {*z12 -= lz;}
}



