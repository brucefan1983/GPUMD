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
class Parameters;
class Dataset;

struct NEP2_Data {
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

class NEP2 : public Potential
{
public:
  struct ParaMB {
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int n_max_radial = 0;       // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0;      // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;              // l = 1, 2, ..., L_max
    int num_Alm_products;
    int num_types = 0;
  };

  struct ANN {
    int dim = 0;          // dimension of the descriptor
    int num_neurons1 = 0; // number of neurons in the hidden layer
    int num_para = 0;     // number of parameters
    const float* w0;      // weight from the input layer to the hidden layer
    const float* b0;      // bias for the hidden layer
    const float* w1;      // weight from the hidden layer to the output layer
    const float* b1;      // bias for the output layer
    const float* c;
  };

  struct ZBL {
    bool enabled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float atomic_numbers[10];
  };

  NEP2(
    char* input_dir,
    Parameters& para,
    int N,
    int N_times_max_NN_radial,
    int N_times_max_NN_angular);
  void
  find_force(Parameters& para, const float* parameters, Dataset& dataset, bool calculate_q_scaler);

private:
  ParaMB paramb;
  ANN annmb;
  NEP2_Data nep_data;
  ZBL zbl;
  void update_potential(const float* parameters, ANN& ann);
};
