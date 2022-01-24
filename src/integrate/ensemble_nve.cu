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
The NVE ensemble integrator.
------------------------------------------------------------------------------*/

#include "ensemble_nve.cuh"

Ensemble_NVE::Ensemble_NVE(int t, int fg)
{
  type = t;
  fixed_group = fg;
}

Ensemble_NVE::~Ensemble_NVE(void)
{
  // nothing now
}

void Ensemble_NVE::compute1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet(
    true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
}

void Ensemble_NVE::compute2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  find_thermo(
    false, box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom,
    thermo);
}
