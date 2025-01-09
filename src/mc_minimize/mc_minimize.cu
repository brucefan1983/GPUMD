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

#include "mc_minimize.cuh"
#include "utilities/read_file.cuh"
#include "mc_minimizer_local.cuh"

void MC_Minimize::parse_group(
  const char** param, int num_param, std::vector<Group>& groups, int num_param_before_group)
{
  if (strcmp(param[num_param_before_group], "group") != 0) {
    PRINT_INPUT_ERROR("invalid option for mc.\n");
  }
  if (!is_valid_int(param[num_param_before_group + 1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method of MC_Minimize should be an integer.\n");
  }
  if (grouping_method < 0) {
    PRINT_INPUT_ERROR("grouping method of MC_Minimize should >= 0.\n");
  }
  if (grouping_method >= groups.size()) {
    PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
  }
  if (!is_valid_int(param[num_param_before_group + 2], &group_id)) {
    PRINT_INPUT_ERROR("group ID of MC_Minimize should be an integer.\n");
  }
  if (group_id < 0) {
    PRINT_INPUT_ERROR("group ID of MC_Minimize should >= 0.\n");
  }
  if (group_id >= groups[grouping_method].number) {
    PRINT_INPUT_ERROR("Group ID should < number of groups.");
  }
}


void MC_Minimize::compute(
  Force& force,
  Atom& atom,
  Box& box,
  std::vector<Group>& group)
{
    mc_minimizer->compute(
    num_trials_mc,
    force,
    atom,
    box,
    group,
    grouping_method,
    group_id);
}

void MC_Minimize::parse_mc_minimize(const char** param, int num_param, std::vector<Group>& group, Atom& atom, Box& box, Force& force)
{
  if (num_param < 6) {
    PRINT_INPUT_ERROR("mc_minimize should have at least 5 parameters.\n");
  }

  //0 for local, 1 for global
  int mc_minimizer_type = 0;
  if (strcmp(param[1], "local") == 0) {
    printf("Perform simple MC with local relaxation:\n");
    mc_minimizer_type = 0;
  } else if (strcmp(param[1], "global") == 0) {
    printf("Perform simple MC with global relaxation:\n");
    mc_minimizer_type = 1;
  } else {
    PRINT_INPUT_ERROR("invalid MC Minimizer type for MC.\n");
  }
  if (mc_minimizer_type == 0) {
    if (num_param < 7) {
      PRINT_INPUT_ERROR("reading error for local relaxation, missing parameter\n");
    }
  }

  //check if num_trials_mc reasonable
  if (!is_valid_int(param[2], &num_trials_mc)) {
    PRINT_INPUT_ERROR("number of MC trials for MC Minimize should be an integer.\n");
  }
  if (num_trials_mc <= 0) {
    PRINT_INPUT_ERROR("number of MC trials for MC Minimize should be positive.\n");
  }

  //check if temperature reasonable
  if (!is_valid_real(param[3], &temperature)) {
    PRINT_INPUT_ERROR("temperature for MC Minimize should be a number.\n");
  }
  if (temperature <= 0) {
    PRINT_INPUT_ERROR("temperature for MC Minimize should be positive.\n");
  }

  //check if force_tolerance reasonable
  if (!is_valid_real(param[4], &force_tolerance)) {
    PRINT_INPUT_ERROR("force tolerance for MC Minimize should be a number.\n");
  }
  if (force_tolerance <= 0) {
    PRINT_INPUT_ERROR("force tolerance for MC Minimize should be positive.\n");
  }

  //check if max nmber of relax reasonable
  if (!is_valid_int(param[5], &max_relax_steps)) {
    PRINT_INPUT_ERROR("max relaxation steps for MC Minimize should be an integer.\n");
  }
  if (max_relax_steps <= 0) {
    PRINT_INPUT_ERROR("max relaxation steps for MC Minimize should be positive.\n");
  }

  //check if scale factor reasonable 
  if (mc_minimizer_type == 0)
  {
  if (!is_valid_real(param[6], &scale_factor)) {
    PRINT_INPUT_ERROR("scale factor for MC Minimize should be a number.\n");
  }
  if (max_relax_steps <= 0) {
    PRINT_INPUT_ERROR("scale factor for MC Minimize should be positive.\n");
  }
  }

  int num_param_before_group;
  if (mc_minimizer_type == 0)
  {
    num_param_before_group = 7;
  }
  if (mc_minimizer_type == 1)
  {
    num_param_before_group = 6;
  }

  if (num_param > num_param_before_group)
  {
    parse_group(param, num_param, group, num_param_before_group);
    printf("    only for atoms in group %d of grouping method %d.\n", group_id, grouping_method);
  }

  if (mc_minimizer_type == 0)
  {
    mc_minimizer.reset(new MC_Minimizer_Local(param, num_param, scale_factor, temperature, force_tolerance, max_relax_steps));
  }
  if (mc_minimizer_type == 1)
  {
    /* code */
  }
  
}