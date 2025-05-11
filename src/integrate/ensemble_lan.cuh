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

class Ensemble_LAN : public Ensemble
{
public:
  Ensemble_LAN();
  Ensemble_LAN(int, int, double, double);
  Ensemble_LAN(int, int, int, int, int, int, int, double, double, double);
  virtual ~Ensemble_LAN(void);

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
  int N_source, N_sink, offset_source, offset_sink;
  double c1, c2, c2_source, c2_sink;
  GPU_Vector<gpurandState> curand_states;
  GPU_Vector<gpurandState> curand_states_source;
  GPU_Vector<gpurandState> curand_states_sink;

  void
  integrate_nvt_lan_half(const GPU_Vector<double>& mass, GPU_Vector<double>& velocity_per_atom);

  void integrate_heat_lan_half(
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    GPU_Vector<double>& velocity_per_atom);
};
