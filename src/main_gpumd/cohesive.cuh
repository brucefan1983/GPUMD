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
#include <vector>

class Box;
class Group;
class Force;

struct D {
  double data[9];
};

class Cohesive
{
public:
  void parse(const char** param, int num_param, int type);
  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    Force& force);

private:
  void parse_cohesive(const char** param, int num_param);
  void parse_elastic(const char** param, int num_param);
  void allocate_memory(const int num_atoms);
  void compute_D();
  void output(Box& box);
  void deform_box(
    const int N, const D& cpu_d, Box& old_box, Box& new_box, GPU_Vector<double>& position_per_atom);
  std::vector<double> cpu_potential_total;
  std::vector<double> cpu_potential_per_atom;
  std::vector<D> cpu_D;
  GPU_Vector<double> new_position_per_atom;
  double strain;
  double start_factor;
  double end_factor;
  double delta_factor;
  int num_points;
  int deformation_type; // 0-7 = cohesive, cubic, hexagonal, trigonal, tetragonal, orthorhombic,
                        // monoclinic, triclinic
};
