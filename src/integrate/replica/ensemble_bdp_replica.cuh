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
The nvt_bdp thermostat of one replica.

Compared with Ensemble_BDP, every step runs on the stream of the replica and
the velocity rescaling is done on the device, so a step needs no host
synchronization. The random-number state, including the cached Gaussian
number, belongs to the replica and can be saved in and restored from a
restart, so that a resumed run continues the same random sequence. Degrees of
freedom removed by a zero total momentum (PRD) can be taken into account.
------------------------------------------------------------------------------*/

#pragma once
#include "integrate/ensemble_bdp.cuh"
#include "integrate/svr_state.cuh"
#include <string>

class Ensemble_BDP_Replica : public Ensemble_BDP
{
public:
  Ensemble_BDP_Replica(const double temperature, const double temperature_coupling);

  // The overloads below run on the stream of the replica.
  using Ensemble_BDP::compute1;
  using Ensemble_BDP::compute2;
  using Ensemble_BDP::find_thermo;

  int removed_degrees_of_freedom = 0;

  // 3 N minus the removed degrees of freedom.
  int get_degrees_of_freedom(const int number_of_atoms) const;

  void compute1(
    const double time_step,
    Atom& atom,
    const gpuStream_t stream);

  void compute2(
    const double time_step,
    const double volume,
    Atom& atom,
    GPU_Vector<double>& thermo,
    const gpuStream_t stream);

  // thermo[0] = temperature, thermo[1] = potential energy, thermo[2..7] = pressure
  // components xx, yy, zz, xy, xz, yz.
  void find_thermo(
    const double volume,
    const Atom& atom,
    GPU_Vector<double>& thermo,
    const gpuStream_t stream);

  // Draws the random numbers of one rescaling exactly as compute2 does, so that
  // batched integration (all replicas of a GPU in one system) keeps each sequence.
  void draw_svr_noise(const int ndeg, double& gaussian, double& sum_of_squares);

  void seed_rng(const unsigned int seed);
  std::string export_rng_state() const;
  bool import_rng_state(const std::string& state);

private:
  SVR_Gaussian_Cache gaussian_cache_;
};
