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
#include "gpu_vector.cuh"
#include "potential.cuh"
class Neighbor;

struct NEP_Data {
  GPU_Vector<double> f12x3b; // 3-body partial forces
  GPU_Vector<double> f12y3b; // 3-body partial forces
  GPU_Vector<double> f12z3b; // 3-body partial forces
  GPU_Vector<int> NN3b;      // 3-body neighbor number
  GPU_Vector<int> NL3b;      // 3-body neighbor list
};

class NEP : public Potential
{
public:
  struct Para2B {
    int num_neurons_per_layer;
    float w0[10];           // from the input layer to the first hidden layer
    float b0[10];           // from the input layer to the first hidden layer
    float w1[100];          // from the first hidden layer to the second hidden layer
    float b1[10];           // from the first hidden layer to the second hidden layer
    float w2[10];           // from the second hidden layer to the output layer
    float b2;               // from the second hidden layer to the output layer
    float r1 = 0.0f;        // inner cutoff
    float r2 = 0.0f;        // outer cutoff
    float pi_factor = 0.0f; // pi/(r2-r1)
  };

  struct Para3B {
    int num_neurons_per_layer;
    float w0[30];           // from the input layer to the first hidden layer
    float b0[10];           // from the input layer to the first hidden layer
    float w1[100];          // from the first hidden layer to the second hidden layer
    float b1[10];           // from the first hidden layer to the second hidden layer
    float w2[10];           // from the second hidden layer to the output layer
    float b2;               // from the second hidden layer to the output layer
    float r1 = 0.0f;        // inner cutoff
    float r2 = 0.0f;        // outer cutoff
    float pi_factor = 0.0f; // pi/(r2-r1)
  };

  NEP(int num_neurons_2b, float r1_2b, float r2_2b, int num_neurons_3b, float r1_3b, float r2_3b);
  void initialize(int, int);
  void update_potential(const float*);
  void find_force(
    int Nc,
    int N,
    int* Na,
    int* Na_sum,
    int max_Na,
    int* type,
    float* h,
    Neighbor* neighbor,
    float* r,
    GPU_Vector<float>& f,
    GPU_Vector<float>& virial,
    GPU_Vector<float>& pe);

private:
  bool has_2b = false;
  bool has_3b = false;

  Para2B para2b;
  Para3B para3b;
  NEP_Data nep_data;
};
