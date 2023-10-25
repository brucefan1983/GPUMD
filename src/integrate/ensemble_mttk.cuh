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

class Ensemble_MTTK : public Ensemble
{
public:
  Ensemble_MTTK(const char** params, int num_params);
  Ensemble_MTTK(void);
  virtual ~Ensemble_MTTK(void);

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

  double t_current = 0, t_start = 0, t_stop = 0, t_target = 0;

protected:
  void init();
  void nhc_temp_integrate();
  void nhc_press_integrate();
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

  enum { NVT, NPT, NPH };
  int ensemble_type = -1;
  bool use_thermostat = false, use_barostat = false;

  enum { NONE, XYZ, XY, YZ, XZ };
  int couple_type = NONE;

  // reset the reference cell for non-hydrostatic barostat
  int h0_reset_interval = 1000;
  // When nph, there is no target temperature. So we use the temperature of kinetic energy.
  double t_for_barostat = 0;
  // the 3x3 matric of cell parameters
  double h[3][3], h_inv[3][3], h_old[3][3];

  // for computing sigma
  double h_old_inv[3][3], h_ref_inv[3][3];
  double vol_ref = 0;

  double tmp1[3][3], tmp2[3][3];
  double sigma[3][3], f_deviatoric[3][3];
  double p_start[3][3], p_stop[3][3], p_current[3][3], p_target[3][3], p_hydro[3][3];
  double p_period[3][3], p_freq[3][3];
  double p_freq_max = 0;
  double omega_dot[3][3], omega_mass[3][3];

  // when applying hydrostatic pressure (iso, aniso, tri),
  // we don't need to compute deviatoric force, since S-p = 0.
  bool non_hydrostatic = false;

  bool p_flag[3][3]; // 1 if control P on this dim, 0 if not
  bool need_scale[3][3];
  double dt, dt2, dt4, dt8, dt16;

  // degrees of freedom when computing temperature
  int temperature_dof = 0;
  double t_freq = 0, t_period = 100;
  double *Q, *eta_dot, *eta_dotdot;
  double *Q_p, *eta_p_dot, *eta_p_dotdot;
  double factor_eta = 0;
  const double kB = 8.617333262e-5;
  // length of Nose-Hoover chain
  int tchain = 4, pchain = 4;
};