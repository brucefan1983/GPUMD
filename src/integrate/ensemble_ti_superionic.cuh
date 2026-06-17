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
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <map>
#include <string>
#include <vector>

enum class SuperionicStage { stage1 = 1, stage2 = 2 };

struct SuperionicUFPair
{
  std::string element_i;
  std::string element_j;
  double p = 0.0;
  double sigma = 0.0;
};

class Ensemble_TI_Superionic : public Ensemble_LAN
{
public:
  Ensemble_TI_Superionic(const char** params, int num_params, SuperionicStage input_stage);
  virtual ~Ensemble_TI_Superionic(void);

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

  void init();
  void find_lambda();
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  SuperionicStage stage;
  FILE* output_file = nullptr;
  double lambda = 0, dlambda = 0;
  int t_equil = -1, t_switch = -1;
  double target_pressure = 0, V = 0, W_forward = 0, W_backward = 0, delta_F = 0;
  bool auto_k = false, has_temperature = false, initialized = false, lambda_active = false;
  std::map<std::string, double> spring_map;
  std::vector<std::string> auto_spring_species;
  std::vector<SuperionicUFPair> uf_pairs;
  std::vector<double> thermo_cpu;
};
