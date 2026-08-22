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

#include "replica.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

void run_replica_md(
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  const std::vector<std::string>& multi_replica_command,
  const std::vector<std::string>& dump_xyz_command,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  const double time_step,
  const int number_of_steps)
{
  if (multi_replica_command.size() < 2 || multi_replica_command[0] != "multi_replica")
    PRINT_INPUT_ERROR("Invalid multi_replica command.");
  int source_device = 0;
  CHECK(gpuGetDevice(&source_device));
  if (multi_replica_command[1] == "remd") {
    replica::run_remd(
      potential_commands,
      ensemble_command,
      multi_replica_command,
      dump_xyz_command,
      source_atom,
      source_box,
      source_group,
      time_step,
      number_of_steps);
  } else if (multi_replica_command[1] == "prd") {
    replica::run_prd(
      potential_commands,
      ensemble_command,
      multi_replica_command,
      dump_xyz_command,
      source_atom,
      source_box,
      source_group,
      time_step,
      number_of_steps);
  } else {
    PRINT_INPUT_ERROR("multi_replica supports remd and prd.");
  }
  CHECK(gpuSetDevice(source_device));
}
