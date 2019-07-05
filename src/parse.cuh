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


void parse_potential_definition(char**, int, Force*);
void parse_potential(char**, int, Force*);
void parse_velocity(char**, int, Atom*);
void parse_ensemble(char**, int, Atom*, Integrate*);
void parse_time_step (char**, int, Atom*);
void parse_neighbor(char**, int, Atom*, Force*);
void parse_dump_thermo(char**, int, Measure*);
void parse_dump_position(char**, int, Measure*, Atom*);
void parse_dump_restart(char**, int, Measure*);
void parse_dump_velocity(char**, int, Measure*);
void parse_dump_force(char**, int, Measure*);
void parse_dump_potential(char**, int, Measure*);
void parse_dump_virial(char**, int, Measure*);
void parse_dump_heat(char**, int, Measure*);
// Helpers for DOS, SDC
void parse_group(char **param, Measure *measure, int *k, Group *group);
void parse_num_dos_points(char **param, Measure *measure, int *k);
//
void parse_compute_dos(char**, int , Measure*, Group *group);
void parse_compute_sdc(char**, int , Measure*, Group *group);
void parse_compute_hac(char**, int , Measure*);
void parse_compute_hnemd(char**, int, Measure*);
void parse_compute_shc(char**,  int, Measure*);
void parse_deform(char**, int, Integrate*);
void parse_compute(char**, int, Measure*);
void parse_fix(char**, int, Atom*);
void parse_run(char**, int, Atom*);
void parse_cutoff(char**, int, Hessian*);
void parse_delta(char**, int, Hessian*);


