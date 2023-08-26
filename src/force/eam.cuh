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
#include "utilities/gpu_vector.cuh"
#include <stdio.h>

#define MAX_NUM_ELEMENTS_EAM 10

struct EAM2004Zhou {
  double re_inv[MAX_NUM_ELEMENTS_EAM];
  double fe[MAX_NUM_ELEMENTS_EAM];
  double rho_e_inv[MAX_NUM_ELEMENTS_EAM];
  double rho_s_inv[MAX_NUM_ELEMENTS_EAM];
  double rho_n[MAX_NUM_ELEMENTS_EAM];
  double rho_n_inv[MAX_NUM_ELEMENTS_EAM];
  double rho_0[MAX_NUM_ELEMENTS_EAM];
  double alpha[MAX_NUM_ELEMENTS_EAM];
  double beta[MAX_NUM_ELEMENTS_EAM];
  double A[MAX_NUM_ELEMENTS_EAM];
  double B[MAX_NUM_ELEMENTS_EAM];
  double kappa[MAX_NUM_ELEMENTS_EAM];
  double lambda[MAX_NUM_ELEMENTS_EAM];
  double Fn0[MAX_NUM_ELEMENTS_EAM];
  double Fn1[MAX_NUM_ELEMENTS_EAM];
  double Fn2[MAX_NUM_ELEMENTS_EAM];
  double Fn3[MAX_NUM_ELEMENTS_EAM];
  double F0[MAX_NUM_ELEMENTS_EAM];
  double F1[MAX_NUM_ELEMENTS_EAM];
  double F2[MAX_NUM_ELEMENTS_EAM];
  double F3[MAX_NUM_ELEMENTS_EAM];
  double eta[MAX_NUM_ELEMENTS_EAM];
  double Fe[MAX_NUM_ELEMENTS_EAM];
  double rc[MAX_NUM_ELEMENTS_EAM];
};

struct EAM2006Dai {
  double A, d, c, c0, c1, c2, c3, c4, B, rc;
};

struct EAM_Data {
  GPU_Vector<double> Fp; // derivative of the density functional
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

class EAM : public Potential
{
public:
  using Potential::compute;
  EAM(FILE*, char*, int num_types, const int number_of_atoms);
  virtual ~EAM(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void initialize_eam2004zhou(FILE*, int num_types);
  void initialize_eam2006dai(FILE*);

protected:
  int potential_model;
  EAM2004Zhou eam2004zhou;
  EAM2006Dai eam2006dai;
  EAM_Data eam_data;
};
