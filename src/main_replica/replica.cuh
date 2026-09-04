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

#include <string>
#include <vector>

class Atom;
class Box;
class Group;

namespace replica
{
void run_remd(
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  const std::vector<std::string>& multi_replica_command,
  const std::vector<std::string>& dump_xyz_command,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  double time_step,
  int number_of_steps);

void run_prd(
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  const std::vector<std::string>& multi_replica_command,
  const std::vector<std::string>& dump_xyz_command,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  double time_step,
  int number_of_steps);
} // namespace replica

void run_replica_md(
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  const std::vector<std::string>& multi_replica_command,
  const std::vector<std::string>& dump_xyz_command,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  double time_step,
  int number_of_steps);
