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
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/read_xyz.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include "utilities/read_file.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace
{
struct Replica_Input {
  std::vector<std::vector<std::string>> potential_commands;
  std::vector<std::string> ensemble_command;
  std::vector<std::string> multi_replica_command;
  std::vector<std::string> dump_xyz_command;
  std::vector<std::string> kspace_command;
  double time_step = 1.0 / TIME_UNIT_CONVERSION;
  int number_of_steps = -1;
};

bool read_command(std::ifstream& input, std::vector<std::string>& command)
{
  while (input.peek() != EOF) {
    const std::vector<std::string> tokens = get_tokens(input);
    command.clear();
    for (const auto& token : tokens) {
      if (token[0] == '#')
        break;
      command.emplace_back(token);
    }
    if (!command.empty())
      return true;
  }
  return false;
}

std::vector<std::vector<std::string>> read_run_input()
{
  std::ifstream input("run.in");
  if (!input.is_open())
    PRINT_INPUT_ERROR("Failed to open run.in.");

  std::vector<std::vector<std::string>> commands;
  std::vector<std::string> command;
  while (read_command(input, command))
    commands.emplace_back(command);
  return commands;
}

Replica_Input parse_run_input(
  const std::vector<std::vector<std::string>>& commands)
{
  Replica_Input input;
  for (size_t command_index = 0; command_index < commands.size(); ++command_index) {
    const std::vector<std::string>& command = commands[command_index];
    const std::string& keyword = command[0];
    if (keyword == "potential") {
      input.potential_commands.push_back(command);
    } else if (keyword == "time_step") {
      double time_step_in_fs = 0.0;
      if (command.size() != 2 ||
          !is_valid_real(command[1].c_str(), &time_step_in_fs) ||
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
      if (!input.kspace_command.empty() || command.size() != 2 ||
          (command[1] != "ewald" && command[1] != "pppm"))
        PRINT_INPUT_ERROR("kspace should be specified once as ewald or pppm.");
      input.kspace_command = command;
    } else if (keyword == "run") {
      if (input.number_of_steps >= 0 || command.size() != 2 ||
          !is_valid_int(command[1].c_str(), &input.number_of_steps) ||
          input.number_of_steps <= 0 || command_index + 1 != commands.size())
        PRINT_INPUT_ERROR(
          "Replica dynamics requires one final run command with a positive number of steps.");
    } else {
      PRINT_INPUT_ERROR(
        "gpumd_replica supports potential, time_step, multi_replica, ensemble, "
        "kspace, dump_xyz, and run commands only.");
    }
  }

  if (input.multi_replica_command.empty() || input.ensemble_command.empty() ||
      input.potential_commands.empty() || input.number_of_steps <= 0)
    PRINT_INPUT_ERROR(
      "Replica input is missing potential, multi_replica, ensemble, or run.");
  return input;
}

void print_welcome_information()
{
  printf("\n");
  printf("***************************************************************\n");
  printf("*                 Welcome to use GPUMD                        *\n");
  printf("*     (Graphics Processing Units Molecular Dynamics)          *\n");
  printf("*                     version 5.7                             *\n");
  printf("*           This is the gpumd_replica executable              *\n");
  printf("***************************************************************\n");
  printf("\n");
}
} // namespace

int main()
{
  print_welcome_information();
  print_compile_information();
  print_gpu_information();

  print_line_1();
  printf("Started running GPUMD replica dynamics.\n");
  print_line_2();

  CHECK(gpuDeviceSynchronize());
  const auto time_begin = std::chrono::high_resolution_clock::now();

  const Replica_Input input = parse_run_input(read_run_input());

  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  print_line_2();
  int has_velocity_in_xyz = 0;
  int number_of_types = 0;
  Atom atom;
  Box box;
  std::vector<Group> group;
  initialize_position(has_velocity_in_xyz, number_of_types, box, group, atom);
  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  print_line_2();

  run_replica_md(
    input.potential_commands,
    input.ensemble_command,
    input.multi_replica_command,
    input.dump_xyz_command,
    atom,
    box,
    group,
    input.time_step,
    input.number_of_steps);

  CHECK(gpuDeviceSynchronize());
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  print_line_1();
  printf("Time used = %f s.\n", time_used.count());
  print_line_2();

  print_line_1();
  printf("Finished running GPUMD replica dynamics.\n");
  print_line_2();
  return EXIT_SUCCESS;
}
