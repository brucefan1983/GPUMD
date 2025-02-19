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
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
class Parameters;
class Dataset;

struct NEP_Data {
  GPU_Vector<int> NN_radial;  // radial neighbor number
  GPU_Vector<int> NL_radial;  // radial neighbor list
  GPU_Vector<int> NN_angular; // angular neighbor number
  GPU_Vector<int> NL_angular; // angular neighbor list
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_radial_factor = 2.5f;
    float typewise_cutoff_angular_factor = 2.0f;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int version = 4; // 3 for NEP3 and 4 for NEP4
    int atomic_numbers[NUM_ELEMENTS];
  };

  struct ANN {
    int dim = 0;                    // dimension of the descriptor
    int num_neurons1 = 0;           // number of neurons in the hidden layer
    int num_para = 0;               // number of parameters
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c;                // for elements in descriptor
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  NEP(
    Parameters& para,
    int N,
    int N_times_max_NN_radial,
    int N_times_max_NN_angular,
    int version,
    int deviceCount);
  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    bool calculate_neighbor,
    int deviceCount);

private:
  ParaMB paramb;
  ANN annmb[16];
  NEP_Data nep_data[16];
  ZBL zbl;
  void update_potential(Parameters& para, float* parameters, ANN& ann);
};
