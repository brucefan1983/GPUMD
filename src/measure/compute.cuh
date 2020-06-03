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

#include "utilities/gpu_vector.cuh"
#include "model/group.cuh"
#include <vector>


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
    int grouping_method = 0;

    void preprocess
    (
        const int N,
        const char* input_dir,
        const std::vector<Group>& group
    );

    void postprocess();
    void process
    (
        const int step,
        const double energy_transferred[],
        const std::vector<Group>& group,
        const GPU_Vector<double>& mass,
        const GPU_Vector<double>& potential_per_atom,
        const GPU_Vector<double>& force_per_atom,
        const GPU_Vector<double>& velocity_per_atom,
        const GPU_Vector<double>& virial_per_atom
    );

private:
    FILE* fid;

    std::vector<double> cpu_group_sum;
    std::vector<double> cpu_group_sum_ave;
    GPU_Vector<double> gpu_group_sum;
    GPU_Vector<double> gpu_per_atom_x;
    GPU_Vector<double> gpu_per_atom_y;
    GPU_Vector<double> gpu_per_atom_z;

    int number_of_scalars = 0;

    void output_results
    (
        const double energy_transferred[],
        const std::vector<Group>& group
    );
};


