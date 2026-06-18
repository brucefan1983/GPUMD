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
  void find_thermo();
  void find_reference_forces(Force& force);
  double get_sum(GPU_Vector<double>& data);
  void accumulate_work();
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  SuperionicStage stage;
  FILE* output_file = nullptr;
  double lambda = 0, dlambda = 0;
  int t_equil = -1, t_switch = -1;
  double target_pressure = 0, V = 0, W_forward = 0, W_backward = 0, delta_F = 0;
  int num_types = 0;
  double beta = 0.0;
  double pe = 0.0;
  double U_einstein = 0.0;
  double U_uf_self = 0.0;
  double U_uf_cross = 0.0;
  double U_aux = 0.0;
  double dHdlambda = 0.0;
  double F_Einstein = 0.0;
  double F_UF_self = 0.0;
  double F_ref = 0.0;
  bool auto_k = false, has_temperature = false, initialized = false, lambda_active = false;
  std::map<std::string, double> spring_map;
  std::vector<std::string> auto_spring_species;
  std::vector<SuperionicUFPair> uf_pairs;
  std::vector<double> thermo_cpu;
  std::vector<double> cpu_k;
  std::vector<double> cpu_spring_mask;
  std::vector<double> cpu_uf_p;
  std::vector<double> cpu_uf_sigma_sqrd;
  std::vector<int> cpu_uf_kind;
  GPU_Vector<double> gpu_k;
  GPU_Vector<double> gpu_spring_mask;
  GPU_Vector<double> gpu_uf_p;
  GPU_Vector<double> gpu_uf_sigma_sqrd;
  GPU_Vector<int> gpu_uf_kind;
  GPU_Vector<double> gpu_einstein;
  GPU_Vector<double> gpu_uf_self;
  GPU_Vector<double> gpu_uf_cross;
  GPU_Vector<double> gpu_aux_fx;
  GPU_Vector<double> gpu_aux_fy;
  GPU_Vector<double> gpu_aux_fz;
  GPU_Vector<double> gpu_cross_fx;
  GPU_Vector<double> gpu_cross_fy;
  GPU_Vector<double> gpu_cross_fz;
  GPU_Vector<double> position_0;

  void prepare_reference_state();
  void validate_species();
  bool is_supported_self_p(double p) const;
  int find_type_for_symbol(const std::string& symbol) const;
  void write_yaml_pair_list(FILE* file, const char* key, bool self_pairs) const;
};
