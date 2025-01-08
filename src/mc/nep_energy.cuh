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
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"

class NEP_Energy
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 0.0f;
    float typewise_cutoff_angular_factor = 0.0f;
    float typewise_cutoff_zbl_factor = 0.0f;
    int version = 4;            // NEP version, 3 for NEP3 and 4 for NEP4
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
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;

  NEP_Energy(void);
  ~NEP_Energy(void);
  void initialize(const char* file_potential);
  void find_energy(
    const int N,
    const int* g_NN_radial,
    const int* g_NN_angular,
    const int* g_type,
    const int* g_t2_radial,
    const int* g_t2_angular,
    const float* g_x12_radial,
    const float* g_y12_radial,
    const float* g_z12_radial,
    const float* g_x12_angular,
    const float* g_y12_angular,
    const float* g_z12_angular,
    float* g_pe);

    //new method

  void find_energy(
    const int N,
    const int* g_NN_radial,
    const int* g_NN_angular,
    const int* g_type,
    const int* g_t2_radial,
    const int* g_t2_angular,
    const float* g_x12_radial,
    const float* g_y12_radial,
    const float* g_z12_radial,
    const float* g_x12_angular,
    const float* g_y12_angular,
    const float* g_z12_angular,
    double* g_pe);

private:
  GPU_Vector<float> nep_parameters; // parameters to be optimized
  void update_potential(float* parameters, ANN& ann);
};
