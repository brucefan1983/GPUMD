/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once
#include "ensemble.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
#include <hiprand/hiprand_kernel.h>
#else
#include <curand_kernel.h>
#endif

class Ensemble_Heat_Hybrid : public Ensemble
{
public:
  Ensemble_Heat_Hybrid(
    int type,
    const std::vector<int>& thermostat_type,
    const std::vector<int>& label,
    const std::vector<int>& size,
    const std::vector<int>& offset,
    double temperature,
    const std::vector<double>& coupling,
    double delta_temperature,
    double time_step);
  virtual ~Ensemble_Heat_Hybrid(void);

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

protected:
  int num_thermostats;
  std::vector<int> thermostat_type;
  std::vector<int> label;
  std::vector<int> size;
  std::vector<int> offset;
  std::vector<double> coupling;
  std::vector<double> c1;
  std::vector<double> c2;
  std::vector<GPU_Vector<gpurandState>> curand_states;

  // Flattened NHC arrays: [thermostat_index * NOSE_HOOVER_CHAIN_LENGTH + chain_index]
  std::vector<double> pos_nhc;
  std::vector<double> vel_nhc;
  std::vector<double> mas_nhc;

  void integrate_heat_hybrid_half(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    GPU_Vector<double>& velocity_per_atom);

  double* get_nhc_pos(int index);
  double* get_nhc_vel(int index);
  double* get_nhc_mas(int index);
  double target_temperature(int index) const;
};
