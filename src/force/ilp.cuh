/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

// TODO: how to pass arguments?
#define MAX_TYPE_ILP 3
#define CUDA_MAX_NL 2048
#define MAX_ILP_NEIGHBOR 3
#define MAX_BIG_ILP_NEIGHBOR 64

struct ILP_Para {
  float rcutsq_ilp[MAX_TYPE_ILP][MAX_TYPE_ILP];
  float d[MAX_TYPE_ILP][MAX_TYPE_ILP];
  float d_Seff[MAX_TYPE_ILP][MAX_TYPE_ILP];      // d / S_R / r_eff
  float C_6[MAX_TYPE_ILP][MAX_TYPE_ILP];
  float z0[MAX_TYPE_ILP][MAX_TYPE_ILP];          // beta
  float lambda[MAX_TYPE_ILP][MAX_TYPE_ILP];      // alpha / beta
  float epsilon[MAX_TYPE_ILP][MAX_TYPE_ILP];
  float C[MAX_TYPE_ILP][MAX_TYPE_ILP];
  float delta2inv[MAX_TYPE_ILP][MAX_TYPE_ILP];   // 1 / delta ^ 2
  float S[MAX_TYPE_ILP][MAX_TYPE_ILP];           // scale
  float rcut_global[MAX_TYPE_ILP][MAX_TYPE_ILP];           // scale
};

struct ILP_Data {
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

class ILP : public Potential
{
public:
  ILP(FILE*, int, int);
  virtual ~ILP(void);
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

protected:
  ILP_Para ilp_para;
  ILP_Data ilp_data;
};

static __constant__ float Tap_coeff[8];