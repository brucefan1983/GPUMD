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

#pragma once
#include "force/force.cuh"
#include "minimizer.cuh"
#include "utilities/common.cuh"

class Minimizer_FIRE : public Minimizer
{
private:
  const double f_inc = 1.1;
  const double f_dec = 0.5;
  const double alpha_start = 0.25;
  const double f_alpha = 0.99;
  const double dt_0 = 1 / TIME_UNIT_CONVERSION; // Time step of 1 fs.
  const double dt_max = 10 * dt_0;
  const double dt_min = 0.02 * dt_0;
  const int N_min = 20;
  const double m = 5; // The mass of atoms. Doesn't matter in minimization.
  GPU_Vector<double> velocity_;
  GPU_Vector<double> temp1_;
  GPU_Vector<double> temp2_;
  GPU_Vector<double> pair_product_;
  GPU_Vector<double> scalar_result_;
  GPU_Pinned_Vector<double> host_scalar_;

  double dot(
    const GPU_Vector<double>& first,
    const GPU_Vector<double>& second,
    gpuStream_t stream);
  void scalar_multiply(
    double factor,
    const GPU_Vector<double>& input,
    GPU_Vector<double>& output,
    gpuStream_t stream);
  void vector_sum(
    const GPU_Vector<double>& first,
    const GPU_Vector<double>& second,
    GPU_Vector<double>& output,
    gpuStream_t stream);

public:
  Minimizer_FIRE(const int number_of_atoms, const int number_of_steps, const double force_tolerance)
    : Minimizer(-1, 0, number_of_atoms, number_of_steps, force_tolerance)
    , velocity_(static_cast<size_t>(number_of_atoms) * 3)
    , temp1_(static_cast<size_t>(number_of_atoms) * 3)
    , temp2_(static_cast<size_t>(number_of_atoms) * 3)
    , pair_product_(static_cast<size_t>(number_of_atoms) * 3)
    , scalar_result_(1)
    , host_scalar_(1, 0.0)
  {
  }

  void compute(
    Force& force,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& position_per_atom,
    std::vector<Group>& group);

  Minimization_Result compute(
    Force& force,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& position_per_atom,
    std::vector<Group>& group,
    gpuStream_t stream,
    bool verbose);
};
