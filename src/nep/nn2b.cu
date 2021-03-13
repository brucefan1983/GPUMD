/*
    Copyright 2019 Zheyong Fan
    This file is part of GPUGA.
    GPUGA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUGA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUGA.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
NN2B: The general 2-body potential based on neural network (NN)
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "gpu_vector.cuh"
#include "mic.cuh"
#include "neighbor.cuh"
#include "nn2b.cuh"

NN2B::NN2B(int num_neurons_per_layer) { para.num_neurons_per_layer = num_neurons_per_layer; };

void NN2B::initialize(int N, int MAX_ATOM_NUMBER)
{
  // nothing
}

void NN2B::update_potential(const float* parameters)
{
  for (int n = 0; n < para.num_neurons_per_layer; ++n) {
    para.w0[n] = parameters[n];
    para.b0[n] = parameters[n + para.num_neurons_per_layer];
    for (int m = 0; m < para.num_neurons_per_layer; ++m) {
      int nm = n * para.num_neurons_per_layer + m;
      para.w1[nm] = parameters[nm + para.num_neurons_per_layer * 2];
    }
    para.b1[n] = parameters[n + para.num_neurons_per_layer * (para.num_neurons_per_layer + 2)];
    para.w2[n] = parameters[n + para.num_neurons_per_layer * (para.num_neurons_per_layer + 3)];
  }
  para.b2 = parameters[para.num_neurons_per_layer * (para.num_neurons_per_layer + 4)];
}

// get U_ij and (d U_ij / d r_ij) / r_ij
static __device__ void find_p2_and_f2(NN2B::Para para, float d12, float& p2, float& f2)
{
  // from the input layer to the first hidden layer
  float x1[10] = {0.0f}; // hidden layer nuerons
  float y1[10] = {0.0f}; // derivatives of the hidden layer nuerons
  for (int n = 0; n < para.num_neurons_per_layer; ++n) {
    x1[n] = tanh(para.w0[n] * d12 / para.r2 - para.b0[n]);
    y1[n] = (1.0f - x1[n] * x1[n]) * para.w0[n] / para.r2;
  }

  // from the first hidden layer to the second hidden layer
  float x2[10] = {0.0f}; // hidden layer nuerons
  float y2[10] = {0.0f}; // derivatives of the hidden layer nuerons
  for (int n = 0; n < para.num_neurons_per_layer; ++n) {
    for (int m = 0; m < para.num_neurons_per_layer; ++m) {
      x2[n] += para.w1[n * para.num_neurons_per_layer + m] * x1[m];
      y2[n] += para.w1[n * para.num_neurons_per_layer + m] * y1[m];
    }
    x2[n] = tanh(x2[n] - para.b1[n]);
    y2[n] *= 1.0f - x2[n] * x2[n];
  }

  // from the hidden layer to the output layer
  for (int n = 0; n < para.num_neurons_per_layer; ++n) {
    p2 += para.w2[n] * x2[n];
    f2 += para.w2[n] * y2[n];
  }
  p2 = para.scaling * (p2 - para.b2);
  f2 *= para.scaling;
}

static __device__ void find_fc_and_fcp(NN2B::Para para, float d12, float& fc, float& fcp)
{
  if (d12 < para.r1) {
    fc = 1.0f;
    fcp = 0.0f;
  } else if (d12 < para.r2) {
    fc = 0.5f * cos(para.pi_factor * (d12 - para.r1)) + 0.5f;
    fcp = -sin(para.pi_factor * (d12 - para.r1)) * para.pi_factor * 0.5f;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static __global__ void find_force_2body(
  int number_of_particles,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  int* g_type,
  NN2B::Para para,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + 18 * blockIdx.x;
    int neighbor_number = g_neighbor_number[n1];

    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];

    float pe = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    float virial_xx = 0.0f;
    float virial_yy = 0.0f;
    float virial_zz = 0.0f;
    float virial_xy = 0.0f;
    float virial_yz = 0.0f;
    float virial_zx = 0.0f;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_neighbor_list[n1 + number_of_particles * i1];

      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

      float p2 = 0.0f, f2 = 0.0f;
      find_p2_and_f2(para, d12, p2, f2);
      float fc, fcp;
      find_fc_and_fcp(para, d12, fc, fcp);
      p2 *= fc;
      f2 = (f2 * fc + p2 * fcp) / d12;

      fx += x12 * f2;
      fy += y12 * f2;
      fz += z12 * f2;
      virial_xx -= x12 * x12 * f2 * 0.5f;
      virial_yy -= y12 * y12 * f2 * 0.5f;
      virial_zz -= z12 * z12 * f2 * 0.5f;
      virial_xy -= x12 * y12 * f2 * 0.5f;
      virial_yz -= y12 * z12 * f2 * 0.5f;
      virial_zx -= z12 * x12 * f2 * 0.5f;
      pe += p2 * 0.5f;
    }

    g_fx[n1] = fx;
    g_fy[n1] = fy;
    g_fz[n1] = fz;
    g_virial[n1 + number_of_particles * 0] = virial_xx;
    g_virial[n1 + number_of_particles * 1] = virial_yy;
    g_virial[n1 + number_of_particles * 2] = virial_zz;
    g_virial[n1 + number_of_particles * 3] = virial_xy;
    g_virial[n1 + number_of_particles * 4] = virial_yz;
    g_virial[n1 + number_of_particles * 5] = virial_zx;
    g_pe[n1] = pe;
  }
}

void NN2B::find_force(
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
  GPU_Vector<float>& pe)
{
  find_force_2body<<<Nc, max_Na>>>(
    N, Na, Na_sum, neighbor->NN, neighbor->NL, type, para, r, r + N, r + N * 2, h, f.data(),
    f.data() + N, f.data() + N * 2, virial.data(), pe.data());
  CUDA_CHECK_KERNEL
}
