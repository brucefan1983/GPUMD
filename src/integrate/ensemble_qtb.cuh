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
#include <vector>
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif

class Ensemble_QTB : public Ensemble
{
public:
  // NVT-QTB constructor
  Ensemble_QTB(int t, int N, double T, double Tc, double dt, double f_max, int N_f, int seed);

  // NPT-QTB constructor (QTB thermostat + Berendsen barostat)
  Ensemble_QTB(
    int t,
    int N,
    double T,
    double Tc,
    double dt,
    double f_max,
    int N_f,
    int seed,
    double target_p[6],
    int num_target_p,
    double pc[6],
    int dx,
    int dy,
    int dz,
    double rate[3]);

  ~Ensemble_QTB(void);

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

private:
  int number_of_atoms;
  int seed;
  int N_f;
  int nfreq2;
  int alpha;
  int counter_mu;

  double dt;
  double h_timestep;
  double fric_coef;
  double f_max_natural;
  double last_filter_temperature;

  std::vector<double> time_H_host;
  GPU_Vector<double> time_H_device;
  GPU_Vector<double> random_array_0;
  GPU_Vector<double> random_array_1;
  GPU_Vector<double> random_array_2;
  GPU_Vector<double> fran;
  GPU_Vector<gpurandState> curand_states;

  void init_qtb_common(int N, double T, double Tc, double dt_input, double f_max_input, int N_f_input, int seed_input);
  void update_time_filter(const double target_temperature);
  void refresh_colored_random_force();
  void apply_qtb_half_step();
};
