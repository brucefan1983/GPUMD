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
#include "nep.cuh"
#include <stdio.h>
#include <vector>

// C H B N
#define MAX_TYPE_ILP_GR_HBN 4
#define CUDA_MAX_NL_GR_HBN 1024
#define MAX_ILP_NEIGHBOR_GR_HBN 3
#define MAX_BIG_ILP_NEIGHBOR_GR_HBN 64

struct ILP_GR_HBN_Para {
  float rcutsq_ilp[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];
  float d[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];
  float d_Seff[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];      // d / S_R / r_eff
  float C_6[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];
  float z0[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];          // beta
  float lambda[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];      // alpha / beta
  float epsilon[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];
  float C[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];
  float delta2inv[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];   // 1 / delta ^ 2
  float S[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];           // scale
  float rcut_global[MAX_TYPE_ILP_GR_HBN][MAX_TYPE_ILP_GR_HBN];           // scale
};

struct ILP_GR_HBN_Data {
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

class ILP_NEP_GR_HBN : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    float typewise_cutoff_zbl_factor = 0.0f;
    int version = 4; // NEP version, 3 for NEP3 and 4 for NEP4
    int model_type =
      0; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
    int num_types = 0;
    float q_scaler[140];
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c;
    // for the scalar part of polarizability
    const float* w0_pol[10];
    const float* b0_pol[10];
    const float* w1_pol[10];
    const float* b1_pol;
  };

  using Potential::compute;
  ILP_NEP_GR_HBN(FILE*, const char*, int, int);
  virtual ~ILP_NEP_GR_HBN(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial){};
  
  virtual void compute_ilp(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    std::vector<Group> &group);
  

private:
  int ilp_group_method;
  double ilp_rc;
#ifdef USE_TABLE
  void construct_table(float* parameters);
#endif

  ParaMB paramb;
  ANN annmb;
  ILP_GR_HBN_Para ilp_para;
  ILP_GR_HBN_Data ilp_data;
  NEP_Data nep_data;
  // rcut for ILP
  double rc_ilp;
  void update_potential(float* parameters, ANN& ann);
};

static __constant__ float Tap_coeff_gr_hbn[8];