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

#pragma once
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Atom
{
public:
  std::vector<int> cpu_type;
  std::vector<int> cpu_type_size;
  std::vector<double> cpu_mass;
  std::vector<double> cpu_position_per_atom;
  std::vector<double> cpu_velocity_per_atom;
  std::vector<std::string> cpu_atom_symbol;
  GPU_Vector<int> type;                  // per-atom type (1 component)
  GPU_Vector<double> mass;               // per-atom mass (1 component)
  GPU_Vector<double> position_per_atom;  // per-atom position (3 components)
  GPU_Vector<double> position_temp;      // used to calculated unwrapped_position
  GPU_Vector<double> unwrapped_position; // unwrapped per-atom position (3 components)
  GPU_Vector<double> velocity_per_atom;  // per-atom velocity (3 components)
  GPU_Vector<double> force_per_atom;     // per-atom force (3 components)
  GPU_Vector<double> heat_per_atom;      // per-atom heat current (5 components)
  GPU_Vector<double> virial_per_atom;    // per-atom virial (9 components)
  GPU_Vector<double> potential_per_atom; // per-atom potential energy (1 component)
  // for beads in PIMD
  GPU_Vector<double> position_beads;
  GPU_Vector<double> velocity_beads;
  GPU_Vector<double> force_beads;
  GPU_Vector<double> potential_beads;
  GPU_Vector<double> virial_beads;
};
