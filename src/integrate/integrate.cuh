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

#include "ensemble.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include <memory>
#include <vector>

class Atom;

class Integrate
{
public:
  std::unique_ptr<Ensemble> ensemble;

  void initialize(
    const int number_of_atoms, const double time_step, const std::vector<Group>& group, Atom& atom);

  void finalize();

  void compute1(
    const double time_step,
    const double step_over_number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  void compute2(
    const double time_step,
    const double step_over_number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  // get inputs from run.in
  void parse_ensemble(Box& box, const char** param, int num_param, std::vector<Group>& group);
  void parse_deform(const char**, int);
  void parse_fix(const char**, int, std::vector<Group>& group);
  void parse_move(const char**, int, std::vector<Group>& group);

  // these data will be used to initialize ensemble
  int type; // ensemble type in a specific run
  int source;
  int sink;
  int fixed_group = -1; // ID of the group in which the atoms will be fixed
  int move_group = -1;  // ID of the group in which the atoms will move with a constant velocity
  double move_velocity[3];

  double temperature;  // target temperature at a specific time
  double temperature1; // target initial temperature for a run
  double temperature2; // target final temperature for a run
  double delta_temperature;
  double target_pressure[6];
  int num_target_pressure_components;
  double temperature_coupling;
  double tau_p;
  double elastic_modulus[6];
  double pressure_coupling[6];
  int deform_x = 0;
  int deform_y = 0;
  int deform_z = 0;
  double deform_rate[3];

  // PIMD
  int number_of_beads;
  int number_of_steps_pimd; // after this number of steps, switch to RPMD
};
