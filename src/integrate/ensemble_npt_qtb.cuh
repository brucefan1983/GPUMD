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

#include "ensemble_mttk.cuh"
#include "utilities/gpu_macro.cuh"
#include <vector>
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif

class Ensemble_NPT_QTB : public Ensemble_MTTK
{
public:
  Ensemble_NPT_QTB(const char** params, int num_params);
  virtual ~Ensemble_NPT_QTB(void);

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
  int qtb_number_of_atoms;
  int qtb_seed;
  int qtb_N_f;
  int qtb_nfreq2;
  int qtb_alpha;
  int qtb_counter_mu;

  double qtb_dt;
  double qtb_h_timestep;
  double qtb_fric_coef;
  double qtb_f_max_natural;
  double qtb_last_filter_temperature;
  double qtb_f_max = 200.0;

  std::vector<double> qtb_time_H_host;
  GPU_Vector<double> qtb_time_H_device;
  GPU_Vector<double> qtb_random_array_0;
  GPU_Vector<double> qtb_random_array_1;
  GPU_Vector<double> qtb_random_array_2;
  GPU_Vector<double> qtb_fran;
  GPU_Vector<gpurandState> qtb_curand_states;

  void init_qtb();
  void qtb_update_time_filter(const double target_temperature);
  void qtb_refresh_colored_random_force();
  void qtb_apply_half_step();

protected:
  virtual void init_mttk() override;
  virtual void get_target_temp() override;
};
