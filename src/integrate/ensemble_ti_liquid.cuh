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
#include <cstdio>
#include <string>
#include <vector>

class Ensemble_TI_Liquid : public Ensemble_LAN
{
public:
  Ensemble_TI_Liquid(const std::vector<std::string>& tokens);
  ~Ensemble_TI_Liquid(void) override;

  void finalize_run(const Atom& atom, const Box& box) override;

  void initialize_before_first_step(
    const double time_step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  void compute2(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo,
    Force& force) override;

  double fe(double x, const double coef[4], const double sum_spline[106], int index);
  void get_UF_sum(const int number_of_atoms);
  void add_UF_force(const Box& box, Atom& atom, Force& force);
  void init(const int number_of_steps, const Atom& atom);
  bool find_lambda(const int step, const int number_of_atoms);
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  FILE* output_file = nullptr;
  double lambda = 0, dlambda = 0;
  int t_equil = -1, t_switch = -1;
  double sigma_sqrd = 1;
  double p = 1;
  double beta;
  double V;
  // The input pressure is only used when reporting the Gibbs free energy.
  double target_pressure = 0;
  double E_diff = 0, E_ref = 0;
  bool auto_switch = true;
  GPU_Vector<double> gpu_eUF;
  GPU_Vector<double> gpu_fx_UF;
  GPU_Vector<double> gpu_fy_UF;
  GPU_Vector<double> gpu_fz_UF;
  GPU_Vector<double> gpu_ti_values;

private:
  void close_output_file(const bool print_message);
};
