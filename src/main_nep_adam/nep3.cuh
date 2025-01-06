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

struct NEP3_Data {
  GPU_Vector<int> NN_radial;  // radial neighbor number
  GPU_Vector<int> NL_radial;  // radial neighbor list
  GPU_Vector<int> NN_angular; // angular neighbor number
  GPU_Vector<int> NL_angular; // angular neighbor list
  GPU_Vector<double> x12_radial;
  GPU_Vector<double> y12_radial;
  GPU_Vector<double> z12_radial;
  GPU_Vector<double> x12_angular;
  GPU_Vector<double> y12_angular;
  GPU_Vector<double> z12_angular;
  GPU_Vector<double> descriptors; // descriptors
  GPU_Vector<double> Fp;          // gradient of descriptors
  GPU_Vector<double> Fp2;         // second gradient of descriptors
  GPU_Vector<double> sum_fxyz;
  GPU_Vector<double> parameters; // parameters to be optimized
};

class NEP3 : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff = false;
    bool use_typewise_cutoff_zbl = false;
    double typewise_cutoff_radial_factor = 2.5;
    double typewise_cutoff_angular_factor = 2.0;
    float typewise_cutoff_zbl_factor = 0.65f;
    double rc_radial = 0.0;     // radial cutoff
    double rc_angular = 0.0;    // angular cutoff
    double rcinv_radial = 0.0;  // inverse of the radial cutoff
    double rcinv_angular = 0.0; // inverse of the angular cutoff
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
    int num_ann = 0;                // number of ANN
    int num_para = 0;               // number of parameters
    const double* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const double* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const double* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const double* b1[10]; // bias for the output layer
    // for the scalar part of polarizability
    const double* w0_pol[10]; // weight from the input layer to the hidden layer
    const double* b0_pol[10]; // bias for the hidden layer
    const double* w1_pol[10]; // weight from the hidden layer to the output layer
    const double* b1_pol[10]; // bias for the output layer
    // for elements in descriptor
    const double* c;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0;
    float rc_outer = 2.0;
    int num_types;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  NEP3(
    Parameters& para,
    int N,
    int N_times_max_NN_radial,
    int N_times_max_NN_angular,
    int version,
    int deviceCount);
  void find_force(
    Parameters& para,
    const double* parameters,
    bool require_grad,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    bool calculate_neighbor,
    int deviceCount);

private:
  ParaMB paramb;
  ANN annmb[16];
  NEP3_Data nep_data[16];
  ZBL zbl;
  void update_potential(Parameters& para, const double* parameters, ANN& ann);
};
