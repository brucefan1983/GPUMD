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

class Ensemble_MSST : public Ensemble
{
public:
  Ensemble_MSST(const char** params, int num_params);
  virtual ~Ensemble_MSST(void);

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

  void remap(double);
  void init();
  void find_thermo();
  void get_omega();
  void get_conserved();
  void get_vsum();
  void msst_v();

  int N;
  int shock_direction;
  double dthalf;
  double vs;
  double qmass;
  double mu;
  double p0;
  double v0;
  double e0;
  double tscale = 0;
  double lagrangian_position = 0;
  double lagrangian_velocity;
  double omega;
  double total_mass = 0;
  double etotal;
  double vsum;
  double ke, temperature;
  double e_conserved, e_msst;
  double vol;
  std::vector<double> thermo_cpu;
  const double kB = 8.617333262e-5;
  GPU_Vector<double> gpu_vsum;
  std::vector<double> cpu_old_velocity;
};
