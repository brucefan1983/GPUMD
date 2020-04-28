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
#include <vector>

class Atom;
class Potential;
class Measure; // TODO: remove this dependence

#define MAX_NUM_OF_POTENTIALS 10

class Force
{
public:

    Force(void);
    ~Force(void);
    void parse_potential_definition(char**, int, Atom*);
    void parse_potential(char**, int);
    void add_potential(char* input_dir, Atom*);
    void compute(Atom*, Measure*);
    int get_number_of_types(FILE *fid_potential);
    void valdiate_potential_definitions(void);
	void initialize_participation_and_shift(Atom*);

    int num_of_potentials;
    std::vector<int> participating_kinds;
    double rc_max;
    int atom_begin[MAX_NUM_OF_POTENTIALS];
    int atom_end[MAX_NUM_OF_POTENTIALS];
    char file_potential[MAX_NUM_OF_POTENTIALS][FILE_NAME_LENGTH];
    std::vector<int> potential_participation;
    std::vector<int> manybody_participation;
    int group_method;
    int num_kind;

private:

    void initialize_potential(char* input_dir, Atom*, int);
    void find_neighbor_local(Atom*, int);
    bool kind_is_participating(int, int);
    bool kinds_are_contiguous(void);

    Potential *potential[MAX_NUM_OF_POTENTIALS];
};


