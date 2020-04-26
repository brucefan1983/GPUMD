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
#include "group.cuh"
#include "neighbor.cuh"
#include "common.cuh"
#include <vector>

const int MAX_NUMBER_OF_GROUPS = 10;

class Atom
{
public:
    int *NN; int *NL;             // global neighbor list
    int *NN_local; int *NL_local; // local neighbor list
    int *type;                    // atom type (for force)
    double *x0; double *y0; double *z0; // determining when to update neighbor list
    double *mass;                   // per-atom mass
    double *x; double *y; double *z;    // per-atom position
    double *vx; double *vy; double *vz; // per-atom velocity
    double *fx; double *fy; double *fz; // per-atom force
    double *heat_per_atom;          // per-atom heat current
    double *virial_per_atom;        // per-atom virial (9 components)
    double *potential_per_atom;     // per-atom potential energy
    double *thermo;                 // some thermodynamic quantities

    std::vector<int> cpu_type;
    std::vector<int> cpu_type_size;
    std::vector<int> shift; // shift to correct type in force eval

    std::vector<double> cpu_mass;
    std::vector<double> cpu_x;
    std::vector<double> cpu_y;
    std::vector<double> cpu_z;
    std::vector<double> cpu_vx;
    std::vector<double> cpu_vy;
    std::vector<double> cpu_vz;

    int N;                // number of atoms 
    int number_of_types;  // number of atom types 

    int has_velocity_in_xyz = 0;
    int num_of_grouping_methods = 0;

    // make a structure?
    int step;
    int number_of_steps; // number of steps in a specific run
    double global_time = 0.0; // run time of entire simulation (fs)
    double initial_temperature; // initial temperature for velocity 
    // time step in a specific run; default value is 1 fs
    double time_step = 1.0 / TIME_UNIT_CONVERSION;

    // some well defined sub-structures
    Neighbor neighbor;
    Box box;
    Group group[MAX_NUMBER_OF_GROUPS];

    Atom(char *input_dir);
    ~Atom(void);

    void initialize_velocity(void);
    void find_neighbor(int is_first);
    void parse_neighbor(char**, int, double);
    void parse_velocity(char**, int);
    void parse_time_step (char**, int);
    void parse_run(char**, int);

private:
    void read_xyz_in_line_1(FILE*);
    void read_xyz_in_line_2(FILE*);
    void read_xyz_in_line_3(FILE*);
    void find_group_size(int);
    void find_group_contents(int);
    void find_type_size(void);
    void initialize_position(char *input_dir);

    void allocate_memory_gpu(void);
    void copy_from_cpu_to_gpu(void);
    void free_memory_gpu(void);

    void find_neighbor_ON2(void);
    void find_neighbor_ON1(int cell_n_x, int cell_n_y, int cell_n_z);
    void find_neighbor(void);
    void check_bound(void);
    int check_atom_distance(void);

    void initialize_velocity_cpu(void);
    void scale_velocity(void);
};


