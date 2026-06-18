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
#include "ensemble_lan.cuh"
#include "force/force.cuh"
#include "langevin_utilities.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <map>
#include <math.h>

class Ensemble_TI_Liquid : public Ensemble_LAN
{
public:
  Ensemble_TI_Liquid(const char** params, int num_params);
  virtual ~Ensemble_TI_Liquid(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo);

  virtual void compute3(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo,
    Force& force);

  void find_thermo();
  double get_UF_sum();
  void add_UF_force(Force& force);
  void init();
  void find_lambda();
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  FILE* output_file;
  double lambda = 0, dlambda = 0;
  int t_equil = -1, t_switch = -1;
  double pe, eUF;
  double sigma_sqrd = 1;
  double p = 1;
  double beta;
  // Force& force;
  //  this is the actual pressure, which may cause problems due to its fluctuation
  double pressure, avg_pressure = 0, V;
  // so I use the input pressure.
  double target_pressure;
  double E_diff = 0, E_ref = 0;
  bool auto_switch = true;
  GPU_Vector<double> gpu_eUF;
  std::vector<double> thermo_cpu;
  // GPU_Vector<int> g_NN;
  // GPU_Vector<int> g_NL;
};
