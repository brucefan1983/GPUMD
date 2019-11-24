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

#include "common.cuh"


class Box
{
public:
    int pbc_x = 1;       // pbc_x = 1 means periodic in the x-direction
    int pbc_y = 1;       // pbc_y = 1 means periodic in the y-direction
    int pbc_z = 1;       // pbc_z = 1 means periodic in the z-direction
    int triclinic = 0;   // triclinic = 1 means the box is non-orthogonal
    real cpu_h[18];
    real get_volume(void);   // get the volume of the box
    void get_inverse(void);  // get the inverse box matrix
};


