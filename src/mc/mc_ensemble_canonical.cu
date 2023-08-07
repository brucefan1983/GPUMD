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
The canonical ensemble for MCMD.
------------------------------------------------------------------------------*/

#include "mc_ensemble_canonical.cuh"

MC_Ensemble_Canonical::MC_Ensemble_Canonical(int num_steps_mc_input, double temperature_input)
{
  num_steps_mc = num_steps_mc_input;
  temperature = temperature_input;
}

MC_Ensemble_Canonical::~MC_Ensemble_Canonical(void)
{
  // nothing now
}

void MC_Ensemble_Canonical::compute(Atom& atom)
{
  for (int step = 0; step < num_steps_mc; ++step) {
    printf("    MC step %d, temperature = %g K.\n", step, temperature);
  }
}
