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

/*----------------------------------------------------------------------------80
The driver class dealing with actions.
------------------------------------------------------------------------------*/

#include "actions.cuh"
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

void Actions::add_persistent_action(Action* action)
{
  persistent_actions.emplace_back(action);
}

void Actions::add_post_force_action(Action* action)
{
  for (Action* post_force_action : post_force_actions) {
    if (post_force_action == action) {
      return;
    }
  }
  post_force_actions.emplace_back(action);
}

void Actions::initialize(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  std::vector<std::string> action_names;
  for (auto& action : run_actions) {
    if (action->action_name == "") {
      printf("Dear developer:\n");
      printf("    Please set the action name you developed.\n");
      exit(1);
    }

    // dump_xyz and dump_netcdf are allowed to be called multiple times; others are not
    if (
      action->action_name != "dump_xyz" && action->action_name != "dump_netcdf") {
      for (auto& action_name : action_names) {
        if (action_name == action->action_name) {
          std::cout << "There are multiple " << action->action_name << " keywords within one run.\n";
          exit(1);
        }
      }
    }
    action_names.emplace_back(action->action_name);
  }


  for (auto& action : run_actions) {
    action->preprocess(
      number_of_steps,
      time_step,
      integrate,
      group,
      atom,
      box,
      force);
  }
}

void Actions::finalize(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{

  for (auto& action : run_actions) {
    action->postprocess(
      atom,
      box,
      integrate,
      number_of_steps,
      time_step,
      temperature);
  }

  for (Action* action : persistent_actions) {
    action->postprocess(
      atom,
      box,
      integrate,
      number_of_steps,
      time_step,
      temperature);
  }

  run_actions.clear();
  post_force_actions.clear();
}

void Actions::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  for (auto& action : run_actions) {
    action->process(
      number_of_steps,
      step,
      fixed_group,
      move_group,
      global_time,
      temperature,
      integrate,
      box,
      group,
      thermo,
      atom,
      force);
  }
}

void Actions::process_dynamics(
  const int md_step,
  Box& box,
  Atom& atom)
{
  for (auto& action : run_actions) {
    action->process_dynamics(md_step, box, atom);
  }
}

void Actions::post_integrate1(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : run_actions) {
    action->post_integrate1(step, time_step, integrate, group, atom, box, force);
  }
}

void Actions::post_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : run_actions) {
    action->post_force(step, time_step, integrate, group, atom, box, force);
  }
  for (Action* action : post_force_actions) {
    action->post_force(step, time_step, integrate, group, atom, box, force);
  }
}

void Actions::post_integrate2(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : run_actions) {
    action->post_integrate2(step, time_step, integrate, group, atom, box, force);
  }
}
