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

class NN2B : public Potential
{
public:
  struct Para {
    int num_neurons_per_layer;
    // from the input layer to the first hidden layer:
    float w0[10];
    float b0[10];
    // from the last hidden layer to the output layer:
    float w1[100];
    float b1[10];
    float w2[10];
    float b2;
    // global scaling
    float scaling = 0.05f;
    float r1 = 0.0f;
    float r2 = 5.0f;
    float pi_factor = 3.1415927f;
  };

  NN2B(int num_neurons_per_layer);
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
  Para para;
};
