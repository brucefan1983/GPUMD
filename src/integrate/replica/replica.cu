/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Entry of replica dynamics, see replica.cuh.
------------------------------------------------------------------------------*/

#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/read_xyz.cuh"
#include "replica.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include "utilities/nep_io/compact_nep.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <cstdio>

namespace
{

replica::Replica_Input parse_run_input(const RunInput& run_input)
{
  replica::Replica_Input input;
  input.time_step = 1.0 / TIME_UNIT_CONVERSION;
  bool has_kspace = false;
  const auto& lines = run_input.lines();
  int last_command = -1;
  for (int n = 0; n < static_cast<int>(lines.size()); ++n) {
    if (!lines[n].tokens.empty())
      last_command = n;
  }
  for (int n = 0; n < static_cast<int>(lines.size()); ++n) {
    std::vector<std::string> command = lines[n].tokens;
    if (command.empty())
      continue;
    const std::string& keyword = command[0];
    if (keyword == "potential") {
      input.potential_commands.push_back(command);
    } else if (keyword == "time_step") {
      double time_step_in_fs = 0.0;
      if (
        command.size() != 2 || !is_valid_real(command[1].c_str(), &time_step_in_fs) ||
        time_step_in_fs <= 0.0)
        PRINT_INPUT_ERROR("time_step should have one positive real parameter.");
      printf("Time step for this run is %g fs.\n", time_step_in_fs);
      input.time_step = time_step_in_fs / TIME_UNIT_CONVERSION;
    } else if (keyword == "multi_replica") {
      if (!input.multi_replica_command.empty())
        PRINT_INPUT_ERROR("Only one multi_replica command is allowed.");
      input.multi_replica_command = command;
    } else if (keyword == "ensemble") {
      if (!input.ensemble_command.empty())
        PRINT_INPUT_ERROR("Only one ensemble command is allowed in a replica run.");
      input.ensemble_command = command;
    } else if (keyword == "dump_xyz") {
      if (!input.dump_xyz_command.empty())
        PRINT_INPUT_ERROR("Only one dump_xyz command is allowed in a replica run.");
      input.dump_xyz_command = command;
    } else if (keyword == "kspace") {
      // The qNEP model reads the kspace command itself. The method and an explicit PPPM
      // spacing are also recorded in restarts, which a resumed run must match.
      if (
        has_kspace || command.size() < 2 || command.size() > 3 ||
        (command[1] != "ewald" && command[1] != "pppm") ||
        (command.size() == 3 && command[1] != "pppm"))
        PRINT_INPUT_ERROR("kspace should be specified once as ewald or pppm [spacing].");
      has_kspace = true;
      input.kspace_method = command[1];
      if (command.size() == 3) {
        double spacing = 0.0;
        if (!is_valid_real(command[2].c_str(), &spacing))
          PRINT_INPUT_ERROR("PPPM spacing should be a number.");
        char text[32];
        snprintf(text, sizeof(text), "%.17g", spacing);
        input.kspace_method += std::string(" ") + text;
      }
    } else if (keyword == "run") {
      if (
        input.number_of_steps >= 0 || command.size() != 2 ||
        !is_valid_int(command[1].c_str(), &input.number_of_steps) ||
        input.number_of_steps <= 0 || n != last_command)
        PRINT_INPUT_ERROR(
          "Replica dynamics requires one final run command with a positive number of steps.");
    } else {
      PRINT_INPUT_ERROR(
        "A multi_replica run supports potential, time_step, multi_replica, ensemble, "
        "kspace, dump_xyz, and run commands only.");
    }
  }
  if (
    input.multi_replica_command.empty() || input.ensemble_command.empty() ||
    input.potential_commands.empty() || input.number_of_steps <= 0)
    PRINT_INPUT_ERROR("Replica input is missing potential, multi_replica, ensemble, or run.");
  if (input.multi_replica_command.size() < 2)
    PRINT_INPUT_ERROR("Invalid multi_replica command.");
  return input;
}

} // namespace

bool has_multi_replica(const RunInput& run_input) { return run_input.contains("multi_replica"); }

void run_replica(const RunInput& run_input)
{
  replica::Replica_Input input = parse_run_input(run_input);

  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  print_line_2();
  int has_velocity_in_xyz = 0;
  int number_of_types = 0;
  Atom atom;
  Box box;
  std::vector<Group> group;
  initialize_position(run_input, has_velocity_in_xyz, number_of_types, box, group, atom);
  // Reading model.xyz prepares the compact NEP files and assigns their atom types.
  input.model_commands = input.potential_commands;
  for (auto& command : input.model_commands) {
    if (command.size() >= 2)
      command[1] = get_compact_nep_filename(command[1]);
  }
  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  print_line_2();

  int source_device = 0;
  CHECK(gpuGetDevice(&source_device));
  if (input.multi_replica_command[1] == "remd") {
    replica::run_remd(input, run_input, atom, box, group);
  } else if (input.multi_replica_command[1] == "prd") {
    replica::run_prd(input, run_input, atom, box, group);
  } else {
    PRINT_INPUT_ERROR("multi_replica supports remd and prd.");
  }
  CHECK(gpuSetDevice(source_device));
}
