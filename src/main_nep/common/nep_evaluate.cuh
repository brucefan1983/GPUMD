/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "utilities/common.cuh"
#include "main_nep/common/nep_compiled_kernels.cuh"
#include "utilities/gpu_vector.cuh"

class Dataset;
class NEP_Neighbor;

struct NEP_Compiled_Evaluation
{
  NEP_Compile library;
  NEP_Compiled_Kernels common;
  using AnnNepFunction =
    int (*)(int, const int*, const float*, const float*, const float*, double*, float*);
  using AnnTemperatureFunction =
    int (*)(int, const int*, const float*, float*, const float*, const float*, double*, float*);
  AnnNepFunction ann_nep_ = nullptr;
  AnnTemperatureFunction ann_temperature_ = nullptr;
  bool valid = false;
  explicit NEP_Compiled_Evaluation(const NEP_Compile_Config& config);
  bool is_valid() const { return valid; }
  void launch_ann(
    int N,
    const int* type,
    const float* descriptors,
    const float* q_scaler,
    const float* parameters,
    double* pe,
    float* Fp);
  void launch_ann_temperature(
    int N,
    const int* type,
    const float* descriptors,
    float* q_scaler,
    const float* temperature,
    const float* parameters,
    double* pe,
    float* Fp);
  void launch_force_radial(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp,
    float* fx, float* fy, float* fz, float* virial);
  void launch_force_angular(
    int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
    const float* x12, const float* y12, const float* z12,
    const float* parameters, const float* Fp, const float* sum_fxyz,
    float* fx, float* fy, float* fz, float* virial);
};

struct NEP_Descriptor_Parameters {
  float rc_radial[NUM_ELEMENTS];  // radial cutoff
  float rc_angular[NUM_ELEMENTS]; // angular cutoff
  int basis_size_radial = 0;
  int basis_size_angular = 0;
  int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
  int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
  int L_max = 0;         // l = 1, 2, ..., L_max
  int dim_angular;
  int has_q_222;
  int has_q_1111;
  int has_q_112;
  int has_q_123;
  int has_q_233;
  int has_q_134;
  int num_L;
  int num_types = 0;
  int num_types_sq = 0;
  int num_c_radial = 0;
  int version = 4;
};

struct NEP_ANN {
  int dim = 0;                    // full ANN input dimension
  int num_neurons1 = 0;           // number of neurons in the hidden layer
  int num_neurons2 = 0;           // number of neurons in the second hidden layer
  int num_hidden_layers = 0;      // number of hidden layers
  int one_ann_no_bias = 0;        // number of parameters in the ANN without bias
  int num_para = 0;               // number of parameters
  const float* wb[NUM_ELEMENTS];  // weights and biases for the hidden layer
  const float* b;                 // bias for the output layer
  const float* c;                 // for elements in descriptor
};

struct NEP_Data {
  GPU_Vector<float> descriptors; // descriptors
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> parameters; // parameters to be optimized
};

void bind_nep_parameters(int num_types, const float* parameters, NEP_ANN& ann);

// Callers select the device and prepare its neighbor list before evaluation.
// N comes from the active dataset; data may have capacity for a larger batch.
// Structural descriptor dimensions are independent of the full ANN dimension.
void compute_nep_descriptors(
  const NEP_Descriptor_Parameters& paramb,
  const float* c,
  const Dataset& dataset,
  const NEP_Neighbor& neighbor,
  NEP_Data& data,
  NEP_Compiled_Evaluation* compiled_kernel);

void evaluate_nep_ann(
  const NEP_ANN& ann,
  const float* temperature,
  Dataset& dataset,
  NEP_Data& data,
  float* q_scaler,
  NEP_Compiled_Evaluation* compiled_kernel);

void compute_nep_forces(
  const NEP_Descriptor_Parameters& paramb,
  const float* c,
  Dataset& dataset,
  const NEP_Neighbor& neighbor,
  NEP_Data& data,
  NEP_Compiled_Evaluation* compiled_kernel);
