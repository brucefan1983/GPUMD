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

class NEP_Charge : public Potential
{
public:
  struct ParaMB {
    int charge_mode = 0;
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
    const float* sqrt_epsilon_inf; // sqrt(epsilon_inf) related to BEC
    const float* b1;               // bias for the output layer
    const float* c;                // for elements in descriptor
  };

  struct NEP_Charge_Data {
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
    GPU_Vector<float> descriptors;       // descriptors
    GPU_Vector<float> charge_derivative; // derivative of charge with respect to descriptor
    GPU_Vector<float> Fp;                // derivative of energy with respect to descriptor
    GPU_Vector<float> sum_fxyz;
    GPU_Vector<float> parameters; // parameters to be optimized
    GPU_Vector<float> kx;
    GPU_Vector<float> ky;
    GPU_Vector<float> kz;
    GPU_Vector<float> G;
    GPU_Vector<float> S_real;
    GPU_Vector<float> S_imag;
    GPU_Vector<float> D_real;
    GPU_Vector<int> num_kpoints;
  };

  struct Charge_Para {
    int num_kpoints_max = 50000;
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float alpha_factor = 1.0f; // 1 / (4 * alpha * alpha)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
    float A;
    float B;
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

  NEP_Charge(
    Parameters& para,
    int N,
    int Nc,
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
  void find_k1k2k3();

private:
  ParaMB paramb;
  ANN annmb[16];
  NEP_Charge_Data nep_data[16];
  ZBL zbl;
  Charge_Para charge_para;
  void update_potential(Parameters& para, float* parameters, ANN& ann);
};
