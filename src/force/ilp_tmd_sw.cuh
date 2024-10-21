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

// MoS2 MoSe2 WSe2 MoSe2
#define MAX_TYPE_ILP_TMD_SW 4
#define CUDA_MAX_NL_TMD 2048
#define MAX_ILP_NEIGHBOR_TMD 6
#define MAX_BIG_ILP_NEIGHBOR_TMD 128
#define MAX_SW_NEIGHBOR_NUM 50

// modification for TMD
#define DELTA1 0.25
#define DELTA2 0.35

struct ILP_TMD_Para {
  float rcutsq_ilp[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];
  float d[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];
  float d_Seff[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];      // d / S_R / r_eff
  float C_6[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];
  float z0[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];          // beta
  float lambda[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];      // alpha / beta
  float epsilon[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];
  float C[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];
  float delta2inv[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];   // 1 / delta ^ 2
  float S[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];           // scale
  float rcut_global[MAX_TYPE_ILP_TMD_SW][MAX_TYPE_ILP_TMD_SW];           // scale
};

struct ILP_TMD_Data {
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

struct SW2_Para {
  // 2-body part
  double A[3][3], B[3][3], a[3][3], sigma[3][3], gamma[3][3], rc[3][3];
  // 3-body part
  double lambda[3][3][3], cos0[3][3][3];
};

struct SW2_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<double> f12x;
  GPU_Vector<double> f12y;
  GPU_Vector<double> f12z;
};

class ILP_TMD_SW : public Potential
{
public:
  using Potential::compute;
  ILP_TMD_SW(FILE*, FILE*, int, int);
  virtual ~ILP_TMD_SW(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    std::vector<Group> &group);
  
  // sw term
  void initialize_sw_1985_1(FILE*); // called by the constructor
  void initialize_sw_1985_2(FILE*); // called by the constructor
  void initialize_sw_1985_3(FILE*); // called by the constructor

protected:
  ILP_TMD_Para ilp_para;
  ILP_TMD_Data ilp_data;
  SW2_Para sw2_para;
  SW2_Data sw2_data;
  // rcut for SW
  double rc_sw;
};

static __constant__ float Tap_coeff_tmd[8];