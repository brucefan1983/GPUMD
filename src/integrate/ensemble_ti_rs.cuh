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
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <math.h>
#include <string>
#include <vector>

class Ensemble_TI_RS : public Ensemble_MTTK
{
public:
  Ensemble_TI_RS(const std::vector<std::string>& tokens);
  ~Ensemble_TI_RS(void) override;

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

  void init(const int number_of_steps, const GPU_Vector<double>& thermo);
  void find_ti_thermo(
    const Box& box,
    const std::vector<Group>& group,
    const Atom& atom,
    GPU_Vector<double>& thermo);
  void scale_force(Atom& atom);
  void find_lambda(
    const int step,
    const Box& box,
    const std::vector<Group>& group,
    const Atom& atom,
    GPU_Vector<double>& thermo);
  double switch_func(double t);
  double dswitch_func(double t);
  void get_target_pressure(
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    const Box& box,
    const Atom& atom,
    GPU_Vector<double>& thermo) override;

protected:
  FILE* output_file = nullptr;
  double lambda_f;
  double lambda = 1, dlambda = 0;
  int t_switch = -1, t_equil = -1;
  double t_max;
  double pe;
  std::vector<double> thermo_cpu;
  bool auto_switch = true;

private:
  void close_output_file(const bool print_message);
};
