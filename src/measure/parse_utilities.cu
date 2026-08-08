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

/*-----------------------------------------------------------------------------------------------100
Functions parsing the options and the per-atom quantities shared by several keywords
--------------------------------------------------------------------------------------------------*/

#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstdio>
#include <cstring>

void parse_group(
  const char** param,
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

void parse_precision(const char** param, const int num_param, int& k, int& precision)
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

static void set_quantity(bool& flag, const char* token, const char* keyword)
{
  if (flag) {
    char message[128];
    snprintf(
      message,
      sizeof(message),
      "Quantity '%s' is specified more than once in %s.\n",
      token,
      keyword);
    PRINT_INPUT_ERROR(message);
  }
  flag = true;
}

bool parse_dump_quantity(
  const char* token,
  DumpQuantities& quantities,
  const bool is_nep_charge,
  const std::vector<Group>& groups,
  const char* keyword)
{
  if (strcmp(token, "velocity") == 0) {
    set_quantity(quantities.has_velocity_, token, keyword);
    printf("    has velocity.\n");
  } else if (strcmp(token, "force") == 0) {
    set_quantity(quantities.has_force_, token, keyword);
    printf("    has force.\n");
  } else if (strcmp(token, "potential") == 0) {
    set_quantity(quantities.has_potential_, token, keyword);
    printf("    has potential.\n");
  } else if (strcmp(token, "unwrapped_position") == 0) {
    set_quantity(quantities.has_unwrapped_position_, token, keyword);
    printf("    has unwrapped position.\n");
  } else if (strcmp(token, "mass") == 0) {
    set_quantity(quantities.has_mass_, token, keyword);
    printf("    has mass.\n");
  } else if (strcmp(token, "charge") == 0) {
    set_quantity(quantities.has_charge_, token, keyword);
    if (is_nep_charge) {
      printf("    has charge predicted by NEP-charge.\n");
    } else {
      printf("    has charge specified in model.xyz.\n");
    }
  } else if (strcmp(token, "bec") == 0) {
    set_quantity(quantities.has_bec_, token, keyword);
    if (is_nep_charge) {
      printf("    has BEC predicted by NEP-charge.\n");
    } else {
      PRINT_INPUT_ERROR("Cannot output BEC for a non-NEP-charge model.\n");
    }
  } else if (strcmp(token, "virial") == 0) {
    set_quantity(quantities.has_virial_, token, keyword);
    printf("    has virial.\n");
  } else if (strcmp(token, "group_labels") == 0) {
    // One column, or one NetCDF grouping_method slot, is written per grouping method. With none
    // the extended XYZ Properties field would say group:I:0, and the NetCDF dimension would have
    // length zero, which NC_UNLIMITED makes indistinguishable from an unlimited dimension.
    if (groups.size() == 0) {
      PRINT_INPUT_ERROR(
        "Cannot output group labels without a grouping method defined in model.xyz.\n");
    }
    set_quantity(quantities.has_group_, token, keyword);
    printf("    has group labels.\n");
  } else {
    return false;
  }
  return true;
}
