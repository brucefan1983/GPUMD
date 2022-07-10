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

class Minimizer
{
public:
  Minimizer(const int number_of_atoms, const int number_of_steps, const double force_tolerance)
    : number_of_atoms_(number_of_atoms),
      number_of_steps_(number_of_steps),
      force_tolerance_(force_tolerance)
  {
    position_per_atom_temp_.resize(number_of_atoms * 3);
    potential_per_atom_temp_.resize(number_of_atoms);
    force_per_atom_temp_.resize(number_of_atoms * 3);

    force_square_max_.resize(1);
    potential_difference_.resize(1);

    cpu_force_square_max_.resize(1);
    cpu_potential_difference_.resize(1);
  }

  virtual ~Minimizer() = default;

  virtual void compute(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom) = 0;

protected:
  void calculate_potential_difference(const GPU_Vector<double>& potential_per_atom);

  void calculate_force_square_max(const GPU_Vector<double>& force_per_atom);

  int number_of_steps_ = 1000;
  int number_of_atoms_ = 0;
  double force_tolerance_ = 1.0e-6;

  GPU_Vector<double> position_per_atom_temp_;
  GPU_Vector<double> potential_per_atom_temp_;
  GPU_Vector<double> force_per_atom_temp_;
  GPU_Vector<double> force_square_max_;
  GPU_Vector<double> potential_difference_;
  std::vector<double> cpu_force_square_max_;
  std::vector<double> cpu_potential_difference_;
};
