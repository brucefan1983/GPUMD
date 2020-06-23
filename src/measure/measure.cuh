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
#include "vac.cuh"
#include "hac.cuh"
#include "shc.cuh"
#include "modal_analysis.cuh"
#include "dump_pos.cuh"
#include "hnemd_kappa.cuh"
#include "compute.cuh"
#include "utilities/gpu_vector.cuh"
#include "model/neighbor.cuh"
#include "model/box.cuh"
#include "model/group.cuh"


class Measure
{
public:

    void initialize
    (
        char* input_dir,
        const int number_of_steps,
        const double time_step,
        const std::vector<Group>& group,
        const std::vector<int>& cpu_type_size,
        const GPU_Vector<double>& mass
    );

    void finalize
    (
        char *input_dir,
        const int number_of_steps,
        const double time_step,
        const double temperature,
        const double volume
    );

    void process
    (
        char *input_dir,
        const int number_of_steps,
        int step,
        const int fixed_group,
        const double global_time,
        const double temperature,
        const double energy_transferred[],
        const std::vector<int>& cpu_type,
        Box& box,
        const Neighbor& neighbor,
        std::vector<Group>& group,
        GPU_Vector<double>& thermo,
        const GPU_Vector<double>& mass,
        const std::vector<double>& cpu_mass,
        GPU_Vector<double>& position_per_atom,
        std::vector<double>& cpu_position_per_atom,
        GPU_Vector<double>& velocity_per_atom,
        std::vector<double>& cpu_velocity_per_atom,
        GPU_Vector<double>& potential_per_atom,
        GPU_Vector<double>& force_per_atom,
        GPU_Vector<double>& virial_per_atom,
        GPU_Vector<double>& heat_per_atom
    );

    int dump_thermo; 
    int dump_velocity;
    int dump_restart;
    int sample_interval_thermo;
    int sample_interval_velocity;
    int sample_interval_restart;
    FILE *fid_thermo;
    FILE *fid_velocity;
    FILE *fid_restart;
    char file_thermo[200];
    char file_velocity[200];
    char file_restart[200];
    VAC vac;
    HAC hac;
    SHC shc;
    HNEMD hnemd;
    Compute compute;
    MODAL_ANALYSIS modal_analysis;
    DUMP_POS* dump_pos = NULL;

    // functions to get inputs from run.in
    void parse_dump_thermo(char**, int);
    void parse_dump_velocity(char**, int);
    void parse_dump_position(char**, int);
    void parse_dump_restart(char**, int);
    void parse_group(char **param, int *k, Group *group);
    void parse_num_dos_points(char **param, int *k);
    void parse_compute_dos(char**, int, Group *group);
    void parse_compute_sdc(char**, int, Group *group);
    void parse_compute_gkma(char**, int, const int number_of_types);
    void parse_compute_hnema(char **, int, const int number_of_types);
    void parse_compute_hac(char**, int);
    void parse_compute_hnemd(char**, int);
    void parse_compute_shc(char**, int, const std::vector<Group>& group);
    void parse_compute(char**, int, const std::vector<Group>& group);

protected:

    void dump_thermos
    (
        FILE *fid,
        const int step,
        const int number_of_atoms,
        const int number_of_atoms_fixed,
        GPU_Vector<double>& gpu_thermo,
        const Box& box
    );

    void dump_velocities
    (
        FILE* fid,
        const int step,
        GPU_Vector<double>& velocity_per_atom,
        std::vector<double>& cpu_velocity_per_atom
    );

    void dump_restarts
    (
        const int step,
        const Neighbor& neighbor,
        const Box& box,
        const std::vector<Group>& group,
        const std::vector<int>& cpu_type,
        const std::vector<double>& cpu_mass,
        GPU_Vector<double>& position_per_atom,
        GPU_Vector<double>& velocity_per_atom,
        std::vector<double>& cpu_position_per_atom,
        std::vector<double>& cpu_velocity_per_atom
    );
};






