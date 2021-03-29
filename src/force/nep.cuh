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

const int MAX_NUM_NEURONS_PER_LAYER = 18;
const int MAX_NUM_N = 9; // n_max+1 = 8+1
const int MAX_NUM_L = 3; // L_max+1 = 2+1
const int MAX_DIM = MAX_NUM_N * MAX_NUM_L;
const int MAX_W0_SIZE = MAX_DIM * MAX_NUM_NEURONS_PER_LAYER;
const int MAX_W1_SIZE = MAX_NUM_NEURONS_PER_LAYER * MAX_NUM_NEURONS_PER_LAYER;

struct NEP_Data {
  GPU_Vector<int> NN3b;       // 3-body neighbor number
  GPU_Vector<int> NL3b;       // 3-body neighbor list
  GPU_Vector<double> f12x;    // 3-body or manybody partial forces
  GPU_Vector<double> f12y;    // 3-body or manybody partial forces
  GPU_Vector<double> f12z;    // 3-body or manybody partial forces
  GPU_Vector<float> Fp;       // derivative of energy with respect to manybody descriptor
  GPU_Vector<float> sum_fxyz; // \sum_k f(n, l, r_{ik}) x_{ik}^a y_{ik}^b z_{ik}^c
};

class NEP : public Potential
{
public:
  struct Para2B {
    float rc = 0.0f;    // cutoff
    float rcinv = 0.0f; // inverse of the cutoff
  };

  struct Para3B {
    float rc = 0.0f;    // cutoff
    float rcinv = 0.0f; // inverse of the cutoff
  };

  struct ParaMB {
    float rc = 0.0f;      // cutoff
    float rcinv = 0.0f;   // inverse of the cutoff
    int n_max = 0;        // n = 0, 1, 2, ..., n_max
    int L_max = 0;        // l = 0, 1, 2, ..., L_max
    float delta_r = 0.0f; // r2/n_max
    float eta = 0.0f;     // \eta in \exp{-\eta * (r - r_i)^2}
  };

  struct ANN {
    int dim = 0;                         // dimension of the descriptor
    int num_neurons_per_layer = 0;       // number of neurons per hidden layer
    float w0[MAX_W0_SIZE];               // weight from the input to the first hidden layer
    float b0[MAX_NUM_NEURONS_PER_LAYER]; // bias for the first hidden layer
    float w1[MAX_W1_SIZE];               // weight from the first to the second hidden layer
    float b1[MAX_NUM_NEURONS_PER_LAYER]; // bias for the second hidden layer
    float w2[MAX_NUM_NEURONS_PER_LAYER]; // weight from the second to the output layer
    float b2;                            // bias for the output layer
  };

  NEP(FILE* fid, const Neighbor& neighbor);
  virtual ~NEP(void);
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
  Para2B para2b;
  Para3B para3b;
  ParaMB paramb;
  ANN ann2b;
  ANN ann3b;
  ANN annmb;
  NEP_Data nep_data;
  void update_potential(FILE* fid);
  void update_potential(const float* parameters, const int offset, ANN& ann);
};
