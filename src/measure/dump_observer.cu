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
Dump energy/force/virial with all loaded potentials at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_observer.cuh"
#include "model/box.cuh"
#include "parse_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>
#include <iostream>

void Dump_Observer::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump observer.\n");

  if (num_param != 3) {
    PRINT_INPUT_ERROR("dump_observer should have 2 parameters.");
  }
  mode_ = param[1];
  if (strcmp(mode_, "observe") != 0 && strcmp(mode_, "average") != 0) {
    PRINT_INPUT_ERROR("observer mode should be 'observe' or 'average'");
  }
  if (!is_valid_int(param[2], &dump_interval_)) {
    PRINT_INPUT_ERROR("observer dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("observer dump interval should > 0.");
  }
  

  printf("    every %d steps.\n", dump_interval_);
}

void Dump_Observer::preprocess(const int number_of_atoms, const int number_of_potentials, Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  dump_exyz_.setup_observer_dump(dump_, dump_interval_, "observer", 1, 1);
  dump_exyz_.preprocess(number_of_atoms, number_of_potentials);
  force.set_multiple_potentials_mode(mode_);
}

void Dump_Observer::process(
    int step,
    const double global_time,
    Box& box,
    Atom& atom,
    Force& force,
    GPU_Vector<double>& thermo)
{
  // Only run if should dump, since forces have to be recomputed with each potential.
  // Note that the dump interval for dump_exyz must be a multiple of dump_observer interval!
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;
  if(strcmp(mode_, "observe") == 0)
  {
    // If observing, calculate properties with all potentials.
    const int number_of_potentials = force.potentials.size();
    for (int potential_index = 0; potential_index < number_of_potentials; potential_index++) {
      force.potentials[potential_index]->compute(box, atom.type, atom.position_per_atom, 
          atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
      dump_exyz_.process(
        step, global_time, box, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
        atom.cpu_position_per_atom, atom.velocity_per_atom, atom.cpu_velocity_per_atom,
        atom.force_per_atom, atom.virial_per_atom, thermo, potential_index);
    }
  }
  else if(strcmp(mode_, "average") == 0)
  {
    // If average, dump already computed properties to file.
    dump_exyz_.process(
      step, global_time, box, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
      atom.cpu_position_per_atom, atom.velocity_per_atom, atom.cpu_velocity_per_atom,
      atom.force_per_atom, atom.virial_per_atom, thermo, 0);
  }
  else {
    PRINT_INPUT_ERROR("Invalid observer mode.\n");
  }
}

void Dump_Observer::postprocess()
{
  dump_ = false;  
}


