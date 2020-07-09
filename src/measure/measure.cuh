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
#include "compute.cuh"
#include "dump_pos.cuh"
#include "dump_restart.cuh"
#include "dump_thermo.cuh"
#include "dump_velocity.cuh"
#include "hac.cuh"
#include "hnemd_kappa.cuh"
#include "modal_analysis.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/neighbor.cuh"
#include "shc.cuh"
#include "utilities/gpu_vector.cuh"
#include "vac.cuh"

class Measure
{
public:
  void initialize(
    char* input_dir,
    const int number_of_steps,
    const double time_step,
    const std::vector<Group>& group,
    const std::vector<int>& cpu_type_size,
    const GPU_Vector<double>& mass);

  void finalize(
    char* input_dir,
    const int number_of_steps,
    const double time_step,
    const double temperature,
    const double volume);

  void process(
    char* input_dir,
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
    GPU_Vector<double>& heat_per_atom);

  VAC vac;
  HAC hac;
  SHC shc;
  HNEMD hnemd;
  Compute compute;
  MODAL_ANALYSIS modal_analysis;
  DUMP_POS* dump_pos = NULL;
  Dump_Velocity dump_velocity;
  Dump_Thermo dump_thermo;
  Dump_Restart dump_restart;

  // functions to get inputs from run.in
  void parse_dump_position(char**, int);
  void parse_group(char** param, int* k, Group* group);
  void parse_num_dos_points(char** param, int* k);
  void parse_compute_dos(char**, int, Group* group);
  void parse_compute_sdc(char**, int, Group* group);
  void parse_compute_gkma(char**, int, const int number_of_types);
  void parse_compute_hnema(char**, int, const int number_of_types);
  void parse_compute_hac(char**, int);
  void parse_compute_hnemd(char**, int);
  void parse_compute_shc(char**, int, const std::vector<Group>& group);
  void parse_compute(char**, int, const std::vector<Group>& group);
};
