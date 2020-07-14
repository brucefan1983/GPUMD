/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

/*-----------------------------------------------------------------------------------------------100
A function parsing the "group" option in some keywords
--------------------------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "parse_group.cuh"
#include "utilities/read_file.cuh"

void parse_group(
  char** param,
  const int num_param,
  const bool allow_all_groups,
  const std::vector<Group>& groups,
  int& k,
  int& grouping_method,
  int& group_id)
{
  if (k + 3 > num_param) {
    PRINT_INPUT_ERROR("Not enough arguments for option 'group'.\n");
  }

  if (!is_valid_int(param[k + 1], &grouping_method)) {
    PRINT_INPUT_ERROR("Grouping method should be an integer.\n");
  }
  if (grouping_method < 0) {
    PRINT_INPUT_ERROR("Grouping method should >= 0.");
  }
  if (grouping_method >= groups.size()) {
    PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
  }

  if (!is_valid_int(param[k + 2], &group_id)) {
    PRINT_INPUT_ERROR("Group ID should be an integer.\n");
  }
  if (group_id >= groups[grouping_method].number) {
    PRINT_INPUT_ERROR("Group ID should < number of groups.");
  }
  if (group_id < 0 && !allow_all_groups) {
    PRINT_INPUT_ERROR("group ID should >= 0.\n");
  }

  printf("    grouping method is %d and group ID is %d.\n", grouping_method, group_id);

  k += 2; // update index for next command
}

void parse_precision(char** param, const int num_param, int& k, int& precision)
{
  if (k + 2 > num_param) {
    PRINT_INPUT_ERROR("Not enough arguments for option 'precision'.\n");
  }
  if (strcmp(param[k + 1], "single") == 0) {
    precision = 1;
    printf("    with single precision.\n");
  } else if (strcmp(param[k + 1], "double") == 0) {
    precision = 2;
    printf("    with double precision.\n");
  } else {
    PRINT_INPUT_ERROR("Invalid precision.\n");
  }
  k++; // update index for next command
}
