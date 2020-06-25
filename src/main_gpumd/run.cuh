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

class Force;
class Integrate;
class Measure;

#include "model/box.cuh"
#include "model/group.cuh"
#include "model/neighbor.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/common.cuh"
#include <vector>


class Run
{
public:
    Run(char*);
private:

    void process_run(char *input_dir);
    void initialize_run();
    void execute_run_in(char* input_dir);
    void parse_one_keyword(char** param, int num_param);
    bool is_velocity;
    bool is_potential;
    bool is_run;
    bool is_potential_definition;

    // keyword parsing functions
    void parse_neighbor(char** param, int num_param);
    void parse_velocity(char** param, int num_param);
    void parse_time_step(char** param, int num_param);
    void parse_run(char** param, int num_param);

    // data in the original Atom class
    int N;                // number of atoms
    int number_of_types;  // number of atom types
    int has_velocity_in_xyz = 0;
    int number_of_steps; // number of steps in a specific run
    double global_time = 0.0; // run time of entire simulation (fs)
    double initial_temperature; // initial temperature for velocity
    double time_step = 1.0 / TIME_UNIT_CONVERSION;
    std::vector<int> cpu_type;
    std::vector<int> cpu_type_size;
    std::vector<double> cpu_mass;
    std::vector<double> cpu_position_per_atom;
    std::vector<double> cpu_velocity_per_atom;
    GPU_Vector<int> type;                  // atom type (for force)
    GPU_Vector<double> mass;               // per-atom mass
    GPU_Vector<double> position_per_atom;  // per-atom position
    GPU_Vector<double> velocity_per_atom;  // per-atom velocity
    GPU_Vector<double> force_per_atom;     // per-atom force
    GPU_Vector<double> heat_per_atom;      // per-atom heat current
    GPU_Vector<double> virial_per_atom;    // per-atom virial (9 components)
    GPU_Vector<double> potential_per_atom; // per-atom potential energy
    GPU_Vector<double> thermo;             // some thermodynamic quantities
    Neighbor neighbor;
    Box box;
    std::vector<Group> group;

    Force force;
    Integrate integrate;
    Measure measure;
};


