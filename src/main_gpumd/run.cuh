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
#include <vector>
#include <iostream>

#ifdef USE_GAS
#include "force/gas-metad.cuh"
#endif

class Run
{
public:
  Run();

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
  Add_Random_Force add_random_force;
  Add_Efield add_efield;
};
