/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <math.h>

class Ensemble_NH : public Ensemble
{
public:
  Ensemble_NH(const char** params, int num_params);
  virtual ~Ensemble_NH(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo);

protected:
  void init();
  void nh_temp_integrate();
  void get_target_temp();
  double get_delta();
  void velocity_verlet_step1();
  void velocity_verlet_step2();
  double find_current_temperature();
  void find_thermo();

  bool tstat_flag, pstat_flag;
  int p_flag[6]; // 1 if control P on this dim, 0 if not
  double dt, dthalf, dt4, dt8;
  double v0, t0;
  int tdof;
  double t_current, t_start, t_stop, t_target;
  double t_freq, t_period;
  double *Q, *eta_dot, *eta_dotdot;
  double p_start[6], p_stop[6], p_target[6];
  double p_period[6], p_freq[6];
  double omega[6], omega_dot[6];
  double omega_mass[6];
  double factor_eta;
  double p_current[6];
  const double kB = 8.617333262e-5;
  int tchain = 4; // length of Nose-Hoover chain
  int pchain = 4;
};
