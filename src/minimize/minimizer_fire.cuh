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
#include "minimizer.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/vector_algo.cuh"

class Minimizer_FIRE : public Minimizer
{
private:
  const double f_inc = 1.1;
  const double f_dec = 0.5;
  const double alpha_start = 0.1;
  const double f_alpha = 0.99;
  const double dt_0 = 1 / TIME_UNIT_CONVERSION; // Time step of 1 fs.
  const double dt_max = 10 * dt_0;
  const int N_min = 5;
  const double m = 5; // The mass of atoms. Doesn't matter in minimization.
  double dt = dt_0;
  double alpha = alpha_start;
  int N_neg = 0;
  double P;

public:
  Minimizer_FIRE(const int number_of_atoms, const int number_of_steps, const double force_tolerance)
    : Minimizer(number_of_atoms, number_of_steps, force_tolerance)
  {
  }

  void compute(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
};