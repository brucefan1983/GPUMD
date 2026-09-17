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
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif
#include <random>
#include <vector>

class Ensemble_PIMD : public Ensemble
{
public:
  Ensemble_PIMD(const char** param, int num_param, const Box& box);

  ~Ensemble_PIMD(void) override;

  void initialize_run(
    const double time_step,
    Atom& atom,
    Box& box,
    const std::vector<Group>& group) override;

  int get_number_of_beads() const
  {
    return number_of_beads;
  }

  double get_temperature1() const
  {
    return temperature1_;
  }

  double get_temperature2() const
  {
    return temperature2_;
  }

  int get_num_target_pressure_components() const
  {
    return num_target_pressure_components;
  }

  void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

protected:
  int number_of_atoms = 0;
  int number_of_beads = 0;
  bool thermostat_internal = false;
  bool thermostat_centroid = false;
  double omega_n;
  bool use_eco_pimd = false;
  bool use_scr_barostat = false;
  bool eco_frequencies_reported = false;
  double eco_omega_max_cm1 = 0.0;
  double eco_last_temperature = -1.0;
  double temperature1_ = 0.0;
  double temperature2_ = 0.0;
  double elastic_modulus_[6] = {0.0};
  double tau_p_ = 0.0;
  GPU_Vector<gpurandState> curand_states;
  GPU_Vector<double*> position_beads;
  GPU_Vector<double*> velocity_beads;
  GPU_Vector<double*> potential_beads;
  GPU_Vector<double*> force_beads;
  GPU_Vector<double*> virial_beads;
  GPU_Vector<double> transformation_matrix;
  GPU_Vector<double> eco_mode_factors;
  GPU_Vector<double> kinetic_energy_virial_part;
  std::vector<double> eco_independent_frequencies;

  GPU_Vector<double> sum_1024; // for intermidiate summation

  void initialize(Atom& atom);
  void update_eco_modes();
  void langevin(const double time_step, Atom& atom);
  std::mt19937 rng;
  void initialize_rng();
};
