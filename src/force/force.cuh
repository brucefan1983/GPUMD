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

#include "potential.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/neighbor.cuh"
#include <memory>
#include <vector>
#include <stdio.h>

#define MAX_NUM_OF_POTENTIALS 10


class Force
{
public:

    Force(void);

    void parse_potential
    (
        char **param, 
        int num_param,
        char* input_dir,
        const Box& box,
        const Neighbor& neighbor,
        const std::vector<int>& cpu_type,
        const std::vector<int>& cpu_type_size
    );

    void add_potential
    (
        char* input_dir,
        const Box& box,
        const Neighbor& neighbor,
        const std::vector<int>& cpu_type,
        const std::vector<int>& cpu_type_size
    );

    void compute
    (
        const Box& box,
        const GPU_Vector<double>& position_per_atom,
        GPU_Vector<int>& type,
        std::vector<Group>& group,
        Neighbor& neighbor,
        GPU_Vector<double>& potential_per_atom,
        GPU_Vector<double>& force_per_atom,
        GPU_Vector<double>& virial_per_atom
    );

    int get_number_of_types(FILE *fid_potential);
    void set_hnemd_parameters(const bool, const double, const double, const double);

    int num_of_potentials;
    double rc_max;
    int atom_begin[MAX_NUM_OF_POTENTIALS];
    int atom_end[MAX_NUM_OF_POTENTIALS];
    bool is_lj[MAX_NUM_OF_POTENTIALS];
    char file_potential[MAX_NUM_OF_POTENTIALS][200];
    int group_method;
    bool compute_hnemd_ = false;
    double hnemd_fe_[3];

private:

    int type_shift_[MAX_NUM_OF_POTENTIALS]; // shift to correct type in force eval

    void initialize_potential
    (
        char* input_dir,
        const Box& box,
        const Neighbor& neighbor,
        const std::vector<int>& cpu_type_size,
        const int m
    );

    void find_neighbor_local
    (
        const int m,
        std::vector<Group>& group,
        GPU_Vector<int>& atom_type,
        const GPU_Vector<double>& position_per_atom,
        const Box& box,
        Neighbor& neighbor
    );

    std::unique_ptr<Potential> potential[MAX_NUM_OF_POTENTIALS];
};


