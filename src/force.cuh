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


class Potential;

class Force
{
public:

    Force(void);      
    ~Force(void);
    void initialize_one_potential(Parameters*, int);
    void initialize(Parameters *para);
    void compute(Parameters*, GPU_Data*);

    int num_of_potentials;
    real rc_max;
    char file_potential[MAX_NUM_OF_POTENTIALS][FILE_NAME_LENGTH];
    Potential *potential[MAX_NUM_OF_POTENTIALS];
    bool build_local_neighbor[MAX_NUM_OF_POTENTIALS];
};




#endif




