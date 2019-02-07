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


#pragma once


// to be deleted
/*
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
*/


static __device__ void dev_apply_mic
(
    int triclinic, int pbc_x, int pbc_y, int pbc_z,
    const real* __restrict__ h, real &x12, real &y12, real &z12
)
{
    if (triclinic == 0) // orthogonal box
    {
        if      (pbc_x == 1 && x12 < - LDG(h,0) * HALF) {x12 += LDG(h,0);}
        else if (pbc_x == 1 && x12 > + LDG(h,0) * HALF) {x12 -= LDG(h,0);}
        if      (pbc_y == 1 && y12 < - LDG(h,1) * HALF) {y12 += LDG(h,1);}
        else if (pbc_y == 1 && y12 > + LDG(h,1) * HALF) {y12 -= LDG(h,1);}
        if      (pbc_z == 1 && z12 < - LDG(h,2) * HALF) {z12 += LDG(h,2);}
        else if (pbc_z == 1 && z12 > + LDG(h,2) * HALF) {z12 -= LDG(h,2);}
    }
    else // triclinic box
    {
        real sx12 = LDG(h,9)  * x12 + LDG(h,10) * y12 + LDG(h,11) * z12;
        real sy12 = LDG(h,12) * x12 + LDG(h,13) * y12 + LDG(h,14) * z12;
        real sz12 = LDG(h,15) * x12 + LDG(h,16) * y12 + LDG(h,17) * z12;
        if (pbc_x == 1) sx12 -= nearbyint(sx12);
        if (pbc_y == 1) sy12 -= nearbyint(sy12);
        if (pbc_z == 1) sz12 -= nearbyint(sz12);
        x12 = LDG(h,0) * sx12 + LDG(h,1) * sy12 + LDG(h,2) * sz12;
        y12 = LDG(h,3) * sx12 + LDG(h,4) * sy12 + LDG(h,5) * sz12;
        z12 = LDG(h,6) * sx12 + LDG(h,7) * sy12 + LDG(h,8) * sz12;
    }
}
