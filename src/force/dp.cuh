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
#include "DeepPot.h"
#include "potential.cuh"
#include <stdio.h>
#include <vector>

namespace deepmd_compat = deepmd;


struct DP_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

// DP neighbor list, which is the same as lammps neighbor list
struct DP_NL {
  int inum;
  int* ilist;
  int* numneigh;
  int** firstneigh;
};

class DP : public Potential
{
public:
  using Potential::compute;
  DP(const char* , int);
  virtual ~DP(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void initialize_dp(const char* filename_dp);

protected:
  // dp coeff
  double ener_unit_cvt_factor;
  double dist_unit_cvt_factor;
  double force_unit_cvt_factor;
  double virial_unit_cvt_factor;
  bool atom_spin_flag;
  bool single_model;

  DP_Data dp_data;
  DP_NL dp_nl;
  // dp instance
  deepmd_compat::DeepPot deep_pot;

  void set_dp_coeff();
};
