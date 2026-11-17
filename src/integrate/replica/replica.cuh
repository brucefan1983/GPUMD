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
Replica dynamics: replica-exchange molecular dynamics (REMD) and parallel
replica dynamics (PRD) of several replicas of the system of model.xyz.

A run.in with a multi_replica command is a replica run. It contains one
potential (energy NEP or qNEP), one ensemble (nvt_bdp, or npt_scr for remd_tp), the multi_replica
command, and one final run command, plus optionally time_step, kspace and
dump_xyz. The replicas are distributed over the visible GPUs; see
integrate/replica/replica_common.cuh for how they are integrated.
------------------------------------------------------------------------------*/

#pragma once
#include <string>
#include <vector>

class Atom;
class Box;
class Group;
class RunInput;

bool has_multi_replica(const RunInput& run_input);
void run_replica(const RunInput& run_input);

namespace replica
{

struct Replica_Input {
  // As in run.in; the potential file named here identifies the model in restarts.
  std::vector<std::vector<std::string>> potential_commands;
  // The same commands with the file actually loaded: for a multi-element NEP model,
  // gpumd evaluates a compact copy restricted to the species of model.xyz, whose type
  // indices are those assigned when model.xyz is read.
  std::vector<std::vector<std::string>> model_commands;
  std::vector<std::string> ensemble_command;
  std::vector<std::string> multi_replica_command;
  std::vector<std::string> dump_xyz_command;
  // "ewald", "pppm" (default spacing) or "pppm <spacing>"
  std::string kspace_method = "pppm";
  double time_step = 0.0; // in natural units
  int number_of_steps = -1;
};

void run_remd(
  const Replica_Input& input,
  const RunInput& run_input,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group);

void run_remd_tp(
  const Replica_Input& input,
  const RunInput& run_input,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group);

void run_prd(
  const Replica_Input& input,
  const RunInput& run_input,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group);

} // namespace replica
