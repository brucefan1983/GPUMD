/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include <string>
#include <vector>

class RunInput;

class Force
{
public:
  Force(void);

  void parse_potential(
    const std::vector<std::string>& tokens,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& mass_per_atom,
    int* position_image = nullptr);

  void finalize();

  int get_number_of_types(FILE* fid_potential);
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
  void set_temperature_range(
    const double temperature1, const double temperature2, const int number_of_steps);
  void advance_temperature();
  int get_number_of_potentials() const;
  Potential& get_potential(const int index);

private:
  std::unique_ptr<Potential> create_potential(
    const std::vector<std::string>& tokens,
    FILE* fid_potential,
    char* potential_name,
    const int num_types,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input,
    bool& is_nep);

  double temperature = 0;
  double delta_T;
  int compute_hnemdec_ = -1;
  double hnemd_fe_[3];
  GPU_Vector<double> coefficient;
  std::vector<std::unique_ptr<Potential>> potentials;
  int number_of_atoms_ = -1;
  bool is_fcp = false;
  bool has_non_nep = false;
  // Workspaces reused by the HNEMDEC heat-flow driving force.
  GPU_Vector<double> hnemdec_tensor_per_atom_;
  GPU_Vector<double> hnemdec_tensor_sum_;
  std::string multiple_potentials_mode_ = "observe"; // "observe" or "average"
  std::string atom_types[NUM_ELEMENTS];

  void check_types(const std::string& file_potential);
  void prepare_compute(
    const int number_of_atoms,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    int* position_image);
  void compute_potentials(
    const int number_of_atoms,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
  void compute_single_potential(
    Potential& potential,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
  void apply_hnemdec(
    const int number_of_atoms,
    GPU_Vector<int>& type,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& mass_per_atom);
  void correct_fcp_force(
    const int number_of_atoms, GPU_Vector<double>& force_per_atom);
};
