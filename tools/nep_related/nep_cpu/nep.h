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
#include <vector>

class NEP3
{
public:
  struct ParaMB {
    int version = 2;            // NEP version, 2 for NEP2 and 3 for NEP3
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int n_max_radial = 0;       // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0;      // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;              // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    float q_scaler[140];
  };

  struct ANN {
    int dim = 0;          // dimension of the descriptor
    int num_neurons1 = 0; // number of neurons in the 1st hidden layer
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

  NEP3(int N);
  void compute(
    const std::vector<int>& NN_radial,
    const std::vector<int>& NL_radial,
    const std::vector<int>& NN_angular,
    const std::vector<int>& NL_angular,
    const std::vector<int>& type,
    const std::vector<float>& r12,
    std::vector<double>& potential_per_atom,
    std::vector<double>& force_per_atom,
    std::vector<double>& virial_per_atom);

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  std::vector<float> Fp;
  std::vector<float> sum_fxyz;
  std::vector<float> parameters;
  void update_potential(const float* parameters, ANN& ann);
};
