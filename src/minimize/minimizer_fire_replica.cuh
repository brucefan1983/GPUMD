/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The FIRE minimizer for the quench of a replica in parallel replica dynamics.

It runs on the stream of the replica with the replica force evaluation, so
that replicas on one GPU can be quenched concurrently, and it returns whether
it converged, the number of steps and the final energy, which decide the
state of the replica.

It derives from Minimizer for its parameters and work buffers. The FIRE loop
is a separate implementation from that of Minimizer_FIRE, which runs on the
default stream with Force: it has the same FIRE update and its own reductions
on the stream; dt and alpha restart at every quench, and the result is
returned.
------------------------------------------------------------------------------*/

#pragma once
#include "minimizer.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_pinned_vector.cuh"

class Replica_Force;

class Minimizer_FIRE_Replica : public Minimizer
{
public:
  struct Result {
    bool converged = false;
    int steps = 0;
    double total_potential = 0.0;
    double force_max = 0.0;
  };

  Minimizer_FIRE_Replica(
    const int number_of_atoms, const int number_of_steps, const double force_tolerance);

  // Not used: a replica is quenched with the replica force evaluation below.
  void compute(
    Force& force,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& position_per_atom,
    std::vector<Group>& group) override;

  Result compute(
    Replica_Force& force, Box& box, Atom& atom, const gpuStream_t stream, const bool verbose);

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
  GPU_Vector<double> pair_product_;
  GPU_Vector<double> scalar_result_;
  GPU_Pinned_Vector<double> host_scalar_;

  double dot(
    const GPU_Vector<double>& first, const GPU_Vector<double>& second, gpuStream_t stream);
  double sum(const GPU_Vector<double>& input, gpuStream_t stream);
  double max_square(const GPU_Vector<double>& input, gpuStream_t stream);
  double scalar_to_host(gpuStream_t stream);
  void scalar_multiply(
    const double factor, const GPU_Vector<double>& input, GPU_Vector<double>& output,
    gpuStream_t stream);
  void vector_sum(
    const GPU_Vector<double>& first,
    const GPU_Vector<double>& second,
    GPU_Vector<double>& output,
    gpuStream_t stream);
};
