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
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "nep_energy.cuh"
#include "../minimize/minimizer_fire.cuh"
#include "utilities/gpu_vector.cuh"
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

class MC_Ensemble
{
public:
  MC_Ensemble(const char** param, int num_param);
  virtual ~MC_Ensemble(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id) = 0;

protected:
  int num_steps_mc = 0;
  double temperature = 0.0;
  std::mt19937 rng;

  std::ofstream mc_output;

  GPU_Vector<int> NN_radial;
  GPU_Vector<int> NN_angular;
  GPU_Vector<int> type_before;
  GPU_Vector<int> type_after;
  GPU_Vector<int> local_type_before;
  GPU_Vector<int> local_type_after;
  GPU_Vector<int> t2_radial_before;
  GPU_Vector<int> t2_radial_after;
  GPU_Vector<int> t2_angular_before;
  GPU_Vector<int> t2_angular_after;
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;
  GPU_Vector<float> pe_before;
  GPU_Vector<float> pe_after;

  NEP_Energy nep_energy;

  bool check_if_small_box(const double rc, const Box& box);
};
