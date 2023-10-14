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

#include "ensemble_ti_spring.cuh"

namespace
{

} // namespace

Ensemble_TI_Spring::Ensemble_TI_Spring(const char** params, int num_params)
{
  use_barostat = false;
  use_thermostat = true;
}

Ensemble_TI_Spring::~Ensemble_TI_Spring(void) {}

void Ensemble_TI_Spring::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  Ensemble_MTTK::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Spring::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  // modify force by spring
  Ensemble_MTTK::compute2(time_step, group, box, atoms, thermo);
}
