/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "spin_integrator.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdint>

class Atom;

class Spin_SIB : public Spin_Integrator
{
public:
  Spin_SIB(
    double alpha,
    double gamma,
    double spin_temperature,
    int seed,
    Atom& atom);

  void compute1(const double time_step, Atom& atom) override;
  void compute2(const double time_step, Atom& atom) override;

private:
  double alpha_;
  double gamma_; // (eV ps)^-1
  double spin_temperature_; // -1: no noise, 0: follow ensemble, >0: fixed K
  std::uint64_t seed_;
  std::uint64_t step_ = 0;
  GPU_Vector<double> spin_start_;
  GPU_Vector<double> noise_increment_;
};
