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

class Force;
class Integrate;
class Measure;

#include "add_efield.cuh"
#include "add_force.cuh"
#include "add_spring.cuh"
#include "add_random_force.cuh"
#include "electron_stop.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "mc/mc.cuh"
#include "measure/measure.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "velocity.cuh"
#include <array>
#include <string>
#include <vector>

class Run
{
public:
  Run(bool skip_run = false, const std::string& run_input_file = "run.in");
  // MDI interface helpers
  int mdi_get_natoms();
  void mdi_get_positions(std::vector<double>& out_positions);
  void mdi_set_positions(const double* positions);
  void mdi_set_forces(const double* forces);
  void mdi_set_energy(double energy);
  void mdi_set_stress(const double* stress_3x3);
  void mdi_compute_forces();
  void mdi_get_forces(std::vector<double>& out_forces);
  void mdi_get_potential(std::vector<double>& out_potential);
  void mdi_initialize_for_mdi();
  void mdi_step_one();
  void mdi_finalize_for_mdi();

private:
  void execute_run_in();
  void perform_a_run();
  void parse_one_keyword(std::vector<std::string>& tokens);

  // keyword parsing functions
  void parse_neighbor(const char** param, int num_param);
  void parse_velocity(const char** param, int num_param);
  void parse_change_box(const char** param, int num_param);
  void parse_correct_velocity(const char** param, int num_param, const std::vector<Group>& group);
  void parse_time_step(const char** param, int num_param);
  void parse_run(const char** param, int num_param);

  // MDI bookkeeping (so dumping/measure works similarly to perform_a_run)
  int mdi_step_counter = 0;
  std::string run_input_file = "run.in";

  bool skip_run_commands = false; // when true, skip 'run' commands (MDI mode)
  bool external_forces_pending =
    false;             // when true, use forces set by mdi_set_forces for next integrate
  bool external_energy_pending = false;
  bool external_stress_pending = false;
  double external_total_energy = 0.0; // eV (system total)
  std::array<double, 9> external_stress = {0.0}; // eV/A^3 (3x3 row-major)

  int number_of_types; // number of atom types
  int has_velocity_in_xyz = 0;
  int number_of_steps;        // number of steps in a specific run
  double global_time = 0.0;   // run time of entire simulation (fs)
  double initial_temperature; // initial temperature for velocity
  double time_step = 1.0 / TIME_UNIT_CONVERSION;
  double max_distance_per_step = -1.0;
  Atom atom;
  GPU_Vector<double> thermo; // some thermodynamic quantities
  Velocity velocity;
  Box box;
  std::vector<Group> group;

  Force force;
  Integrate integrate;
  MC mc;
  Measure measure;
  Electron_Stop electron_stop;
  Add_Force add_force;
  Add_Spring add_spring;
  Add_Random_Force add_random_force;
  Add_Efield add_efield;
};
