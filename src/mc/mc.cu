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

/*----------------------------------------------------------------------------80
The driver class for the various MC ensembles.
------------------------------------------------------------------------------*/

#include "mc.cuh"
#include "mc_ensemble_canonical.cuh"
#include "mc_ensemble_sgc.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"

void MC::initialize(void)
{
  // todo
}

void MC::finalize(void) { do_mcmd = false; }

void MC::compute(int step, int num_steps, Atom& atom, Box& box, std::vector<Group>& group)
{
  if (do_mcmd) {
    if ((step + 2) % num_steps_md == 0) {
      double temperature =
        temperature_initial + step * (temperature_final - temperature_initial) / num_steps;
      mc_ensemble->compute(step + 2, temperature, atom, box, group, grouping_method, group_id);
    }
  }
}

void MC::parse_mc(
  const char** param, int num_param, std::vector<Group>& groups, std::vector<int>& cpu_type)
{
  if (num_param < 6) {
    PRINT_INPUT_ERROR("mc should have at least 5 parameters.\n");
  }

  int mc_ensemble_type = 0;
  if (strcmp(param[1], "canonical") == 0) {
    mc_ensemble_type = 0;
  } else if (strcmp(param[1], "sgc") == 0) {
    mc_ensemble_type = 1;
  } else if (strcmp(param[1], "vcsgc") == 0) {
    PRINT_INPUT_ERROR(
      "variance constrained semi-grand canonical MCMD has not been implemented yet.\n");
  } else {
    PRINT_INPUT_ERROR("invalid MC ensemble for MCMD.\n");
  }

  if (!is_valid_int(param[2], &num_steps_md)) {
    PRINT_INPUT_ERROR("number of MD steps for MCMD should be an integer.\n");
  }
  if (num_steps_md <= 0) {
    PRINT_INPUT_ERROR("number of MD steps for MCMD should be positive.\n");
  }

  if (!is_valid_int(param[3], &num_steps_mc)) {
    PRINT_INPUT_ERROR("number of MC steps for MCMD should be an integer.\n");
  }
  if (num_steps_mc <= 0) {
    PRINT_INPUT_ERROR("number of MC steps for MCMD should be positive.\n");
  }

  if (!is_valid_real(param[4], &temperature_initial)) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be a number.\n");
  }
  if (temperature_initial <= 0) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be positive.\n");
  }

  if (!is_valid_real(param[5], &temperature_final)) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be a number.\n");
  }
  if (temperature_final <= 0) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be positive.\n");
  }

  if (mc_ensemble_type == 0) {
    if (num_param > 6) {
      if (num_param != 9) {
        PRINT_INPUT_ERROR("mc canonical must has 9 paramters when using a grouping method.\n");
      }
      if (strcmp(param[6], "group") != 0) {
        PRINT_INPUT_ERROR("invalid option for mc.\n");
      }
      if (!is_valid_int(param[7], &grouping_method)) {
        PRINT_INPUT_ERROR("grouping method of MCMD should be an integer.\n");
      }
      if (grouping_method < 0) {
        PRINT_INPUT_ERROR("grouping method of MCMD should >= 0.\n");
      }
      if (grouping_method >= groups.size()) {
        PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
      }
      if (!is_valid_int(param[8], &group_id)) {
        PRINT_INPUT_ERROR("group ID of MCMD should be an integer.\n");
      }
      if (group_id < 0) {
        PRINT_INPUT_ERROR("group ID of MCMD should >= 0.\n");
      }
      if (group_id >= groups[grouping_method].number) {
        PRINT_INPUT_ERROR("Group ID should < number of groups.");
      }

      bool has_multi_types = false;
      int type0 = 0;
      for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
        int n =
          groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
        if (k == 0) {
          type0 = cpu_type[n];
        } else {
          if (cpu_type[n] != type0) {
            has_multi_types = true;
            break;
          }
        }
      }
      if (!has_multi_types) {
        PRINT_INPUT_ERROR("Must have more than one atom type in the specified group.");
      }
    }
  }

  if (mc_ensemble_type == 0) {
    printf("Perform canonical MCMD:\n");
    mc_ensemble.reset(new MC_Ensemble_Canonical(num_steps_mc));
  } else if (mc_ensemble_type == 1) {
    printf("Perform SGC MCMD:\n");
    mc_ensemble.reset(new MC_Ensemble_SGC(num_steps_mc));
  }
  printf("    after every %d MD steps, do %d MC trials.\n", num_steps_md, num_steps_mc);
  printf(
    "    with an initial temperature of %g K and a final temperature of %g K.\n",
    temperature_initial,
    temperature_final);

  if (grouping_method < 0) {
    printf("    for all the atoms in the system.\n");
  } else {
    printf("    only for atoms in group %d of grouping method %d.\n", group_id, grouping_method);
  }

  do_mcmd = true;
}