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
#include "box.cuh"
#include "utilities/gpu_vector.cuh"


class Neighbor
{
public:
    int MN;                // upper bound of # neighbors for one particle
    int update;            // 1 means you want to update the neighbor list
    int number_of_updates; // number of updates during a run
    double skin;             // skin distance 
    double rc;               // cutoff used when building the neighbor list

    GPU_Vector<int> NN, NL;             // global neighbor list
    GPU_Vector<int> NN_local, NL_local; // local neighbor list

    // some data for the ON1 method
    GPU_Vector<int> cell_count;
    GPU_Vector<int> cell_count_sum;
    GPU_Vector<int> cell_contents;

    // used to determine when to update neighbor list
    GPU_Vector<double> x0, y0, z0;

    void find_neighbor(int, const Box&, GPU_Vector<double>&);

private:
    void find_neighbor_ON2(const Box&, double*, double*, double*);
    void find_neighbor_ON1(int, int, int, const Box&, double*, double*, double*);
    void find_neighbor(const Box&, double*, double*, double*);
    void check_bound(void);
    int check_atom_distance(double* x, double* y, double* z);
};


