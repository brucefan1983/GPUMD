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
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include <math.h>

class Ensemble_MSST : public Ensemble
{
public:
  Ensemble_MSST(const char** params, int num_params);

  void initialize_before_first_step(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

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

  void remap(const double dilation, Box& box, Atom& atom);
  void init(
    const Box& box,
    const std::vector<Group>& group,
    Atom& atom,
    GPU_Vector<double>& thermo);
  void find_thermo(
    const Box& box,
    const std::vector<Group>& group,
    const Atom& atom,
    GPU_Vector<double>& thermo);
  void get_omega();
  void get_conserved(
    const Box& box,
    const std::vector<Group>& group,
    const Atom& atom,
    GPU_Vector<double>& thermo);
  void get_vsum(const Atom& atom);
  void msst_v(Atom& atom);

  int N;
  int shock_direction;
  double dthalf;
  double vs;
  double qmass;
  double mu;
  double p0;
  double v0;
  double e0;
  bool p0_given = false;
  bool v0_given = false;
  bool e0_given = false;
  double p_current;
  double dhugo, dray;
  double tscale = 0;
  double lagrangian_position = 0;
  double lagrangian_velocity;
  double omega;
  double total_mass = 0;
  double etotal;
  double ke, temperature;
  double e_conserved, e_msst;
  double vol;
  const double kB = 8.617333262e-5;
  std::vector<double> thermo_cpu;
  GPU_Vector<double> gpu_vsum;
  GPU_Vector<double> gpu_v_backup;
};
