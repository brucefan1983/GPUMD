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
#include "gpu_vector.cuh"


struct Neighbor
{
    int MN;                // upper bound of # neighbors for one particle
    int update;            // 1 means you want to update the neighbor list
    int number_of_updates; // number of updates during a run
    double skin;             // skin distance 
    double rc;               // cutoff used when building the neighbor list

    // some data for the ON1 method
    GPU_Vector<int> cell_count;
    GPU_Vector<int> cell_count_sum;
    GPU_Vector<int> cell_contents;
};


