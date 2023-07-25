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
#include "utilities/common.cuh"
#include <memory>
#include <stdio.h>
#include <vector>

// TODO: include ILP
#include "ilp.cuh"

class Force
{
public:
  Force(void);

  void
  parse_potential(const char** param, int num_param, const Box& box, const int number_of_atoms);

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
  void set_multiple_potentials_mode(std::string mode);

  bool compute_hnemd_ = false;
  int compute_hnemdec_ = -1;
  double hnemd_fe_[3];
  double temperature;
  GPU_Vector<double> coefficient;
  std::vector<std::unique_ptr<Potential>> potentials;

private:
  int number_of_atoms_ = -1;
  bool is_fcp = false;
  bool has_non_nep = false;
  std::string multiple_potentials_mode_ = "observe"; // "observe" or "average"
  std::string atom_types[NUM_ELEMENTS];

  void check_types(const char* file_potential);
};
