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

class Force;
class Hessian;

#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/neighbor.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Phonon
{
public:
  Phonon(char*);

private:
  void compute(char*, Force&, Hessian&);
  void parse(char**, int, Force&, Hessian&, char* input_dir);

  // data in the original Atom class
  int N;               // number of atoms
  int number_of_types; // number of atom types
  int has_velocity_in_xyz = 0;
  Atom atom;
  GPU_Vector<double> thermo; // some thermodynamic quantities
  Neighbor neighbor;
  Box box;
  std::vector<Group> group;
};
