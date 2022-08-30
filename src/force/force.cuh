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

#include "model/box.cuh"
#include "model/group.cuh"
#include "potential.cuh"
#include <memory>
#include <stdio.h>
#include <vector>

#define MAX_NUM_OF_POTENTIALS 10

class Force
{
public:
  Force(void);

  void parse_potential(
    char** param,
    int num_param,
    char* input_dir,
    const Box& box,
    const std::vector<int>& cpu_type,
    const std::vector<int>& cpu_type_size);

  void add_potential(
    char* input_dir,
    const Box& box,
    const std::vector<int>& cpu_type,
    const std::vector<int>& cpu_type_size);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& mass_per_atom);

  int get_number_of_types(FILE* fid_potential);
  void set_hnemd_parameters(const bool, const double, const double, const double);
  void set_hnemdec_parameters(
    const int compute_hnemdec,
    const double hnemd_fe_x,
    const double hnemd_fe_y,
    const double hnemd_fe_z,
    const std::vector<double>& mass,
    const std::vector<int>& type,
    const std::vector<int>& type_size,
    const double T);

  int num_of_potentials;
  double rc_max;
  int num_types[MAX_NUM_OF_POTENTIALS];
  int atom_begin[MAX_NUM_OF_POTENTIALS];
  int atom_end[MAX_NUM_OF_POTENTIALS];
  bool is_lj[MAX_NUM_OF_POTENTIALS];
  char file_potential[MAX_NUM_OF_POTENTIALS][200];
  int group_method;
  bool compute_hnemd_ = false;
  int compute_hnemdec_ = 0;
  double hnemd_fe_[3];
  double temperature;
  GPU_Vector<double> coefficient;

private:
  bool is_fcp = false;
  int type_shift_[MAX_NUM_OF_POTENTIALS]; // shift to correct type in force eval

  void initialize_potential(
    char* input_dir,
    const Box& box,
    const int num_atoms,
    const std::vector<int>& cpu_type_size,
    const int m);

  std::unique_ptr<Potential> potential[MAX_NUM_OF_POTENTIALS];
};
