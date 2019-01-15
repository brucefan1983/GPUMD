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




class Compute
{
public:
    int compute_temperature = 0;
    int compute_potential = 0;
    int compute_force = 0;
    int compute_virial = 0;
    int compute_jp = 0;
    int compute_jk = 0;

    int sample_interval = 1;
    int output_interval = 1;
    int use_new_group = 0;

    void preprocess(char*, Atom*);
    void postprocess(Atom* atom, Integrate*);
    void process(int, Atom*, Integrate*);

private:
    FILE* fid;

    real* cpu_group_sum;
    real* cpu_group_sum_ave;
    real* gpu_group_sum;
    real* gpu_per_atom_x;
    real* gpu_per_atom_y;
    real* gpu_per_atom_z;

    int number_of_scalars = 0;

    void output_results(Atom*, Integrate*);
};




