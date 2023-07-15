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

// TODO
#define MAX_TYPE_ILP 10

struct ILP_Para {
  double r_cut[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double d_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double S_R_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double r_eff_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double C_6_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double a_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double b_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double e_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double C_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
  double g_ij[MAX_TYPE_ILP][MAX_TYPE_ILP];
};

struct ILP_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
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
  void initialize_ilp(FILE* fid, int, const std::vector<int>, int);

protected:
  ILP_Para ilp_para;
  ILP_Data ilp_data;
};
