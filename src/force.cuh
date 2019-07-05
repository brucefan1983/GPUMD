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
#include <list>
#include <vector>

using namespace std;

#define MAX_NUM_OF_POTENTIALS 10

class Force
{
public:

    Force(void);
    ~Force(void);
    void add_potential(Atom*);
    void compute(Atom*, Measure*);
    int get_number_of_types(FILE *fid_potential);

    int num_of_potentials;
    int* participating_kinds;
    real rc_max;
    int atom_begin[MAX_NUM_OF_POTENTIALS];
    int atom_end[MAX_NUM_OF_POTENTIALS];
    char file_potential[MAX_NUM_OF_POTENTIALS][FILE_NAME_LENGTH];

    vector<list<int>> interaction_pairs;
    int* manybody_definition;
    int group_method;
    int num_kind;

private:

    void initialize_potential(Atom*, int);
    void find_neighbor_local(Atom*, int);

    Potential *potential[MAX_NUM_OF_POTENTIALS];
};


