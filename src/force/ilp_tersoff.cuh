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
#include "potential.cuh"
#include <stdio.h>
#include <vector>
#include "utilities/gpu_vector.cuh"
#include "tersoff1988.cuh"

// C B N
#define MAX_TYPE_ILP_TERSOFF 3
#define CUDA_MAX_NL_CBN 2048
#define MAX_ILP_NEIGHBOR_CBN 3
#define MAX_BIG_ILP_NEIGHBOR_CBN 128
#define MAX_TERSOFF_NEIGHBOR_NUM 50

struct ILP_CBN_Para {
  float rcutsq_ilp[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];
  float d[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];
  float d_Seff[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];      // d / S_R / r_eff
  float C_6[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];
  float z0[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];          // beta
  float lambda[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];      // alpha / beta
  float epsilon[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];
  float CC[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];           // C
  float delta2inv[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];   // 1 / delta ^ 2
  float S[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];           // scale
  float rcut_global[MAX_TYPE_ILP_TERSOFF][MAX_TYPE_ILP_TERSOFF];           // scale
};

struct ILP_CBN_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> reduce_NL;
  GPU_Vector<int> big_ilp_NN, big_ilp_NL;
  GPU_Vector<int> ilp_NN, ilp_NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<float> f12x;
  GPU_Vector<float> f12y;
  GPU_Vector<float> f12z;
  GPU_Vector<float> f12x_ilp_neigh;
  GPU_Vector<float> f12y_ilp_neigh;
  GPU_Vector<float> f12z_ilp_neigh;
};

class ILP_TERSOFF : public Potential
{
public:
  using Potential::compute;
  ILP_TERSOFF(FILE*, FILE*, int, int);
  virtual ~ILP_TERSOFF(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  
  virtual void compute_ilp(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    std::vector<Group> &group);
  
  void initialize_tersoff_1988(FILE*, int num_atoms);

protected:
  ILP_CBN_Para ilp_para;
  ILP_CBN_Data ilp_data;
  Tersoff1988_Data tersoff_data;
  GPU_Vector<double> ters;
  int ilp_group_method = 0;
  // rcut for Tersoff potential
  double rc_tersoff;
  int num_types;
};

static __constant__ float Tap_coeff_CBN[8];