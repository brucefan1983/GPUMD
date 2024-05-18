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
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
class Parameters;
class Dataset;

struct NEP5_Data {
  GPU_Vector<int> NN;  // neighbor number
  GPU_Vector<int> NL;  // neighbor list
  GPU_Vector<float> x12;
  GPU_Vector<float> y12;
  GPU_Vector<float> z12;
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP5 : public Potential
{
public:
  struct ParaMB {
    float rc = 0.0f;     // cutoff
    float rcinv = 0.0f;  // inverse of the cutoff
    int basis_size = 0;
    int n_max = 0;       // n = 0, 1, 2, ..., n_max
    int L_max = 0;       // l = 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
  };

  struct ANN {
    int dim = 0;          // dimension of the descriptor
    int num_neurons1 = 0; // number of neurons in the hidden layer
    int num_para = 0;     // number of parameters
    const float* w0[100]; // weight from the input layer to the hidden layer
    const float* b0[100]; // bias for the hidden layer
    const float* w1[100]; // weight from the hidden layer to the output layer
    const float* b1;      // bias for the output layer
    // for the scalar part of polarizability
    const float* w0_pol[10]; // weight from the input layer to the hidden layer
    const float* b0_pol[10]; // bias for the hidden layer
    const float* w1_pol[10]; // weight from the hidden layer to the output layer
    const float* b1_pol;     // bias for the output layer
    // for elements in descriptor
    const float* c;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types;
    float para[550];
    float atomic_numbers[NUM_ELEMENTS];
  };

  NEP5(
    Parameters& para,
    int N,
    int N_times_max_NN,
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
  NEP5_Data nep_data[16];
  ZBL zbl;
  void update_potential(Parameters& para, float* parameters, ANN& ann);
};
