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

#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Atom;
class Force;
class Integrate;

class Dump_Polarizability
{
public:
  void parse(const char** param, int num_param);
  void preprocess(const int number_of_atoms, const int number_of_potentials, Force& force);
  void process(
    int step,
    const double global_time,
    const int number_of_atoms_fixed,
    std::vector<Group>& group,
    Box& box,
    Atom& atom,
    Force& force);
  void postprocess();

private:
  bool dump_ = false;
  int dump_interval_ = 1;
  FILE* file_;
  GPU_Vector<double> gpu_pol_;
  std::vector<double> cpu_pol_;
  void write_polarizability(const int step);
  Atom atom_copy;
};
