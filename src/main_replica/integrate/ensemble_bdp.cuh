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
#include "ensemble.cuh"
#include "svr_state.cuh"
#include <random>
#include <string>

struct BDP_RNG_State {
  std::mt19937 rng;
  SVR_Gaussian_Cache gaussian_cache;
};

class Ensemble_BDP : public Ensemble
{
public:
  Ensemble_BDP(int, int, double*, double, double);
  Ensemble_BDP(int, int, int, double, double, double);
  virtual ~Ensemble_BDP(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  void compute1_stream(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo,
    const gpuStream_t stream);

  void compute2_stream(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo,
    const gpuStream_t stream);

  void seed_rng(unsigned int seed);
  BDP_RNG_State get_rng_state() const;
  void set_rng_state(const BDP_RNG_State& state);
  std::string export_rng_state() const;
  bool import_rng_state(const std::string& state);

protected:
  std::mt19937 rng;
  SVR_Gaussian_Cache gaussian_cache;
  void initialize_rng();

  void integrate_nvt_bdp_2(
    const double time_step,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo);

  void integrate_nvt_bdp_2(
    const double time_step,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo,
    const gpuStream_t stream);

  void integrate_heat_bdp_2(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);
};
