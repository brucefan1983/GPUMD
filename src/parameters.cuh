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



// Parameters for neighbor list updating
struct Neighbor
{
    int MN;               // upper bound of # neighbors for one particle
    int update;           // 1 means you want to update the neighbor list
    real skin;            // skin distance 
    real rc;              // cutoff used when building the neighbor list
};




// Parameters in the code (in a mess)
struct Parameters 
{
    // a structure?
    int N;                // number of atoms
    int number_of_groups; // number of groups 
    int fixed_group;      // ID of the group in which the atoms will be fixed 
    int number_of_types;  // number of atom types 

    // a structure?
    int pbc_x;           // pbc_x = 1 means periodic in the x-direction
    int pbc_y;           // pbc_y = 1 means periodic in the y-direction
    int pbc_z;           // pbc_z = 1 means periodic in the z-direction

    // make a structure?
    int number_of_steps; // number of steps in a specific run
    real initial_temperature; // initial temperature for velocity
    real temperature1;
    real temperature2; 
    // time step in a specific run; default value is 1 fs
    real time_step;

    // some well defined sub-structures
    Neighbor neighbor;
};




