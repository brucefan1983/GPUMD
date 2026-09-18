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

#include "ensemble.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include <memory>
#include <string>
#include <vector>

class Atom;

class Integrate
{
private:
  std::unique_ptr<Ensemble> ensemble_;

  EnsembleType type = EnsembleType::UNKNOWN;
  int fixed_group = -1; // ID of the group in which the atoms will be fixed
  int move_group = -1;  // ID of the group in which the atoms will move with a constant velocity
  int fixed_grouping_method = 0;
  int move_grouping_method = 0;
  double move_velocity[3];

  double temperature1; // target initial temperature for a run
  double temperature2; // target final temperature for a run
  int num_target_pressure_components;
  int deform_x = 0;
  int deform_y = 0;
  int deform_z = 0;
  int deform_xy = 0;
  int deform_xz = 0;
  int deform_yz = 0;

  // PIMD
  int number_of_beads;

public:
  bool has_ensemble() const;
  EnsembleType get_type() const;
  int get_fixed_group() const;
  int get_move_group() const;
  int get_fixed_grouping_method() const;
  int get_move_grouping_method() const;
  double get_temperature1() const;
  double get_temperature2() const;
  int get_num_target_pressure_components() const;
  int get_number_of_beads() const;
  const double* get_energy_transferred() const;
  const std::vector<double>& get_energy_transferred_n() const;
  void find_thermo(
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& velocity_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& thermo);
  void set_deform(
    int deform_x,
    int deform_y,
    int deform_z,
    int deform_xy,
    int deform_xz,
    int deform_yz);

  void initialize(
    double time_step,
    Atom& atom,
    Box& box,
    const std::vector<Group>& group);

  void finalize(const Atom& atom, const Box& box);

  void compute1(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  void compute2(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo,
    Force& force);

  // get inputs from run.in
  void parse_ensemble(
    const std::vector<std::string>& tokens,
    const Atom& atom,
    const Box& box,
    const std::vector<Group>& group);
  void parse_fix(const std::vector<std::string>& tokens, const std::vector<Group>& group);
  void parse_move(const std::vector<std::string>& tokens, const std::vector<Group>& group);

  // Kept public for the optional PLUMED integration.
  double temperature;  // target temperature at a specific time
};
