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
#include "vac.cuh"
#include "dos.cuh"
#include "sdc.cuh"
#include "hac.cuh"
#include "shc.cuh"
#include "gkma.cuh"
#include "hnema.cuh"
#include "dump_pos.cuh"
#include "hnemd_kappa.cuh"
#include "compute.cuh"


class Measure
{
public:
    Measure(char *input_dir);
    ~Measure(void);
    void initialize(char*, Atom*);
    void finalize(char*, Atom*, Integrate*);
    void process(char*, Atom*, Integrate*, int);
    int dump_thermo; 
    int dump_restart;
    int dump_velocity;
    int dump_force;
    int dump_potential;
    int dump_virial;
    int dump_heat;
    int sample_interval_thermo;
    int sample_interval_restart;
    int sample_interval_velocity;
    int sample_interval_force;
    int sample_interval_potential;
    int sample_interval_virial;
    int sample_interval_heat;
    FILE *fid_thermo;
    FILE *fid_restart;
    FILE *fid_velocity;
    FILE *fid_force;
    FILE *fid_potential;
    FILE *fid_virial;
    FILE *fid_heat;
    char file_thermo[FILE_NAME_LENGTH];   
    char file_restart[FILE_NAME_LENGTH];
    char file_velocity[FILE_NAME_LENGTH];
    char file_force[FILE_NAME_LENGTH];
    char file_potential[FILE_NAME_LENGTH];
    char file_virial[FILE_NAME_LENGTH];
    char file_heat[FILE_NAME_LENGTH];
    VAC vac;
    DOS dos;
    SDC sdc;
    HAC hac;
    SHC shc;
    HNEMD hnemd;
    Compute compute;
    GKMA gkma;
    HNEMA hnema;
    DUMP_POS* dump_pos;

    // functions to get inputs from run.in
    void parse_dump_thermo(char**, int);
    void parse_dump_position(char**, int, Atom*);
    void parse_dump_restart(char**, int);
    void parse_dump_velocity(char**, int);
    void parse_dump_force(char**, int);
    void parse_dump_potential(char**, int);
    void parse_dump_virial(char**, int);
    void parse_dump_heat(char**, int);
    void parse_group(char **param, int *k, Group *group);
    void parse_num_dos_points(char **param, int *k);
    void parse_compute_dos(char**, int, Group *group);
    void parse_compute_sdc(char**, int, Group *group);
    void parse_compute_gkma(char**, int, Atom*);
    void parse_compute_hnema(char **, int, Atom*);
    void parse_compute_hac(char**, int);
    void parse_compute_hnemd(char**, int);
    void parse_compute_shc(char**, int);
    void parse_compute(char**, int);

protected:
    void dump_thermos(FILE*, Atom*, Integrate*, int);
    void dump_restarts(Atom*, int);
    void dump_velocities(FILE*, Atom*, int);
    void dump_forces(FILE*, Atom*, int);
    void dump_potentials(FILE*, Atom*, int);
    void dump_virials(FILE*, Atom*, int);
    void dump_heats(FILE*, Atom*, int);
};


