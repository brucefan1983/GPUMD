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
#include "utilities/common.cuh"
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
  void get_target_pressure();
  double get_delta();
  void get_sigma();
  double find_current_temperature();
  void find_current_pressure();
  void find_thermo();
  void get_h_matrix_from_box();
  void copy_h_matrix_to_box();
  void get_p_hydro();
  void get_deviatoric();
  void nh_omega_dot();
  void propagate_box();
  void propagate_box_off_diagonal();
  void propagate_box_diagonal();
  void scale_positions();
  void nh_v_press();
  void couple();

  enum { NONE, XYZ, XY, YZ, XZ };
  int couple_type = NONE;
  int h0_reset_interval = 1000;
  double h[3][3], h_inv[3][3], h_old[3][3], h_old_inv[3][3], h_ref_inv[3][3];
  double tmp1[3][3], tmp2[3][3];
  double sigma[3][3], fdev[3][3];
  double p_start[3][3], p_stop[3][3], p_current[3][3], p_target[3][3], p_hydro[3][3];
  double p_period[3][3], p_freq[3][3];
  double omega[3][3], omega_dot[3][3], omega_mass[3][3];
  bool tstat_flag = false, pstat_flag = false, deviatoric_flag = false;
  bool p_flag[3][3]; // 1 if control P on this dim, 0 if not
  double dt, dthalf, dt4, dt8, dt16;
  double v0, t0;
  int tdof;
  double t_current, t_start, t_stop, t_target;
  double t_freq, t_period;
  double *Q, *eta_dot, *eta_dotdot;
  double factor_eta;
  const double kB = 8.617333262e-5;
  int tchain = 4; // length of Nose-Hoover chain
  int pchain = 4;
};
