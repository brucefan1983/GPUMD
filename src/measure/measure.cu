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
The driver class dealing with measurement.
------------------------------------------------------------------------------*/

#include "measure.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

void Measure::initialize(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& prop : properties) {
    prop->preprocess(
      number_of_steps,
      time_step,
      integrate,
      group,
      atom,
      box,
      force);
  }

  const int number_of_atoms = atom.mass.size();
#ifdef USE_NETCDF
  dump_netcdf.preprocess(number_of_atoms);
#endif
#ifdef USE_PLUMED
  plmd.preprocess(atom.cpu_mass);
#endif
}

void Measure::finalize(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature,
  const double number_of_beads)
{

  for (auto& prop : properties) {
    prop->postprocess(
      atom,
      box,
      integrate,
      number_of_steps,
      time_step,
      temperature,
      number_of_beads);
  }

  properties.clear();

#ifdef USE_NETCDF
  dump_netcdf.postprocess();
#endif
#ifdef USE_PLUMED
  plmd.postprocess();
#endif
}

void Measure::process(
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

  for (auto& prop : properties) {
    prop->process(
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

#ifdef USE_NETCDF
  dump_netcdf.process(
    step,
    global_time,
    box,
    atom.cpu_type,
    atom.position_per_atom,
    atom.cpu_position_per_atom,
    atom.velocity_per_atom,
    atom.cpu_velocity_per_atom);
#endif
}
