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
class RunInput;

#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "velocity.cuh"
#include <string>
#include <vector>

class Run
{
public:
  Run(const RunInput& run_input);

private:
  void execute_run_in(const RunInput& run_input);
  void perform_a_run(const int number_of_steps);
  void compute_force();
  void parse_one_keyword(
    const std::vector<std::string>& tokens, const RunInput& run_input);

  // keyword parsing functions
  void parse_velocity(const std::vector<std::string>& tokens);
  void parse_change_box(const std::vector<std::string>& tokens);
  void parse_correct_velocity(
    const std::vector<std::string>& tokens, const std::vector<Group>& group);
  void parse_time_step(const std::vector<std::string>& tokens);
  void parse_run(const std::vector<std::string>& tokens);

  int number_of_types; // number of atom types
  int has_velocity_in_xyz = 0;
  bool has_seen_dftd3_command = false;
  bool has_seen_kspace_command = false;
  bool has_replicate_ = false;
  int replicate_size_[3] = {1, 1, 1};
  std::string first_potential_filename_;
  double global_time = 0.0; // run time of entire simulation (fs)
  double time_step = 1.0 / TIME_UNIT_CONVERSION;
  double max_distance_per_step = -1.0;
  Atom atom;
  GPU_Vector<double> thermo; // some thermodynamic quantities
  Velocity velocity;
  Box box;
  std::vector<Group> group;

  Force force;
  Integrate integrate;
  Measure measure;
};
