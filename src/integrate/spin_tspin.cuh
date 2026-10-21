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
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "spin_integrator.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Atom;

class Spin_TSPIN : public Spin_Integrator
{
public:
  Spin_TSPIN(
    double initial_temperature,
    double temperature_coupling,
    double time_step,
    const std::vector<double>& mass_factor_by_type,
    int seed,
    Atom& atom);

  void compute1(const double time_step, Atom& atom) override;
  void compute2(const double time_step, Atom& atom) override;

private:
  double find_twice_kinetic_energy(const Atom& atom);
  double find_nhc_factor(const double time_step, const Atom& atom);
  void initialize_spin_velocity(
    double initial_temperature,
    int seed,
    Atom& atom);

  double pos_nhc_[4];
  double vel_nhc_[4];
  double mas_nhc_[4];
  int number_of_active_atoms_ = 0;
  std::vector<double> cpu_mass_factor_per_atom_;
  GPU_Vector<double> mass_factor_per_atom_;
  GPU_Vector<double> twice_kinetic_energy_;
};
