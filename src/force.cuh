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




#ifndef FORCE1_H
#define FORCE1_H


#define MAX_NUM_OF_POTENTIALS 10
#define FILE_NAME_LENGTH      200


class Potential;
class Measure;

class Force
{
public:

    Force(void);      
    ~Force(void);
    void initialize_one_potential(Parameters*, int);
    void initialize_two_body_potential(Parameters*);
    void initialize_many_body_potential(Parameters*, CPU_Data*, int);
    void initialize(char*, Parameters *para, CPU_Data*, Atom*);
    void find_neighbor_local(Parameters*, Atom*, int);
    void compute(Parameters*, Atom*, Measure*);

    int num_of_potentials;
    int interlayer_only;
    real rc_max;
    char file_potential[MAX_NUM_OF_POTENTIALS][FILE_NAME_LENGTH];
    Potential *potential[MAX_NUM_OF_POTENTIALS];
    int type_begin[MAX_NUM_OF_POTENTIALS];
    int type_end[MAX_NUM_OF_POTENTIALS];
    int *layer_label;
};




#endif




