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
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"

void MC::initialize(void)
{
  // todo
}

void MC::finalize(void) { do_mcmd = false; }

void MC::compute(int step, Atom& atom, Box& box)
{
  if (do_mcmd) {
    if ((step + 2) % num_steps_md == 0) {
      mc_ensemble->compute(step + 2, atom, box);
    }
  }
}

void MC::parse_cmc(const char** param, int num_param)
{
  if (num_param != 4) {
    PRINT_INPUT_ERROR("cmc should have 3 parameter.\n");
  }

  if (!is_valid_int(param[1], &num_steps_md)) {
    PRINT_INPUT_ERROR("number of MD steps for cmc should be an integer.\n");
  }
  if (num_steps_md <= 0) {
    PRINT_INPUT_ERROR("number of MD steps for cmc should be positive.\n");
  }

  if (!is_valid_int(param[2], &num_steps_mc)) {
    PRINT_INPUT_ERROR("number of MC steps for cmc should be an integer.\n");
  }
  if (num_steps_mc <= 0) {
    PRINT_INPUT_ERROR("number of MC steps for cmc should be positive.\n");
  }

  if (!is_valid_real(param[3], &temperature)) {
    PRINT_INPUT_ERROR("temperature for cmc should be a number.\n");
  }
  if (temperature <= 0) {
    PRINT_INPUT_ERROR("temperature for cmc should be positive.\n");
  }

  printf("Perform canonical MC:\n");
  printf("    after every %d MD steps, do %d MC trials.\n", num_steps_md, num_steps_mc);
  printf("    with a temperature of %g K.\n", temperature);

  mc_ensemble.reset(new MC_Ensemble_Canonical(num_steps_mc, temperature));

  do_mcmd = true;
}