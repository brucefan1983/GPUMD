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
class Neighbor;

struct NEP2_Data {
  GPU_Vector<double> f12x; // 3-body or manybody partial forces
  GPU_Vector<double> f12y; // 3-body or manybody partial forces
  GPU_Vector<double> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN;
  GPU_Vector<int> NL;
  GPU_Vector<float> atomic_number;
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
    int L_max = 0;              // l = 0, 1, 2, ..., L_max
    float q_scaler[91];
    float q_min[91];
  };

  struct ANN {
    int dim = 0;          // dimension of the descriptor
    int num_neurons1 = 0; // number of neurons in the 1st hidden layer
    int num_neurons2 = 0; // number of neurons in the 2nd hidden layer
    int num_para = 0;     // number of parameters
    const float* w0;      // weight from the input to the first hidden layer
    const float* b0;      // bias for the first hidden layer
    const float* w1;      // weight from the first to the second hidden layer
    const float* b1;      // bias for the second hidden layer
    const float* w2;      // weight from the second to the output layer
    const float* b2;      // bias for the output layer
  };

  NEP2(FILE* fid, char* input_dir, const Neighbor& neighbor);
  virtual ~NEP2(void);
  virtual void compute(
    const int type_shift,
    const Box& box,
    const Neighbor& neighbor,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  ParaMB paramb;
  ANN annmb;
  NEP2_Data nep_data;
  void update_potential(FILE* fid);
  void update_potential(const float* parameters, ANN& ann);
};
