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

/*----------------------------------------------------------------------------80
The k-space part of the PPPM method.
------------------------------------------------------------------------------*/

#include "pppm.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <vector>
#include <iostream>

namespace{

int get_best_K(int m)
{
  int n = 16;
  while (n < m) {
    n *= 2;
  }
  return n;
}

void cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] =  a[1] * b [2] - a[2] * b [1];
  c[1] =  a[2] * b [0] - a[0] * b [2];
  c[2] =  a[0] * b [1] - a[1] * b [0];
}

__constant__ float sinc_coeff[6] = {1.0f, -1.6666667e-1f, 8.3333333e-3f, -1.9841270e-4f, 2.7557319e-6f, -2.5052108e-8f};

__device__ inline float sinc(float x)
{
  float sinc = 0.0f;
  if (x * x <= 1.0f) {
    float term = 1.0f;
    for (int i = 0; i < 6; ++i) {
      sinc += sinc_coeff[i] * term;
      term *= x * x;
    }
  } else {
    sinc = sin(x) / x;
  }
  return sinc;
}

void __global__ find_k_and_G_opt(
  const PPPM::Para para,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    int nk[3];
    nk[2] = n / para.K0K1;
    nk[1] = (n - nk[2] * para.K0K1) / para.K[0];
    nk[0] = n % para.K[0];

    // Eq. (6.40) in Allen & Tildesley
    float denominator[3] = {0.0f};
    for (int d = 0; d < 3; ++d) {
      //if (nk[d] >= para.K_half[d]) {
        //nk[d] -= para.K[d];
      //}
      denominator[d] = sin(0.5f * para.two_pi_over_K[d] * nk[d]);
      denominator[d] *= denominator[d];
      denominator[d] = 1.0f - denominator[d] + 0.13333333f * denominator[d] * denominator[d];
      denominator[d] *= denominator[d];
    }
    float kx = nk[0] * para.b[0][0] + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    float ky = nk[0] * para.b[0][1] + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    float kz = nk[0] * para.b[0][2] + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    g_kx[n] = kx;
    g_ky[n] = ky;
    g_kz[n] = kz;
    float ksq = kx * kx + ky * ky + kz * kz;

    // Eq. (6.39) in Allen & Tildesley
    float numerator = sinc(0.5f * para.two_pi_over_K[0] * nk[0]);
    numerator *= sinc(0.5f * para.two_pi_over_K[1] * nk[1]);
    numerator *= sinc(0.5f * para.two_pi_over_K[2] * nk[2]);
    numerator *= numerator * numerator;
    numerator *= numerator;

    // Eq. (41) in Allen & Tildesley
    float G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
    G_opt /= denominator[0] * denominator[1] * denominator[2];

    //if (nk[0] * nk[1] * nk[2] != 0) {
    if (n != 0) {
      g_G[n] = G_opt;
    } else {
      g_G[n] = 0.0;
    }
  }
}

__device__ inline int get_index_within_mesh(const int K, const int n)
{
  if (n >= K) {
    return n - K;
  } else if (n < 0) {
    return n + K;
  }
}

__global__ void find_charge_mesh(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  float* g_charge_mesh)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    float q = g_charge[n];
    float sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    float sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    float sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    float reduced_pos[3] = {sx * para.K[0], sy * para.K[1], sz * para.K[2]};
    int ix = int(reduced_pos[0] + 0.5); // can be 0, ..., K[0]
    int iy = int(reduced_pos[1] + 0.5); // can be 0, ..., K[1]
    int iz = int(reduced_pos[2] + 0.5); // can be 0, ..., K[2]
    float dx = reduced_pos[0] - ix; // (-0.5, 0.5)
    float dy = reduced_pos[1] - iy; // (-0.5, 0.5)
    float dz = reduced_pos[2] - iz; // (-0.5, 0.5)
    // Eq. (6.29) in Allen & Tildesley
    float Wx[3] = {0.5f * (0.5f - dx) * (0.5f - dx), 0.75f - dx * dx, 0.5f * (0.5f + dx) * (0.5f + dx)};
    float Wy[3] = {0.5f * (0.5f - dy) * (0.5f - dy), 0.75f - dy * dy, 0.5f * (0.5f + dy) * (0.5f + dy)};
    float Wz[3] = {0.5f * (0.5f - dz) * (0.5f - dz), 0.75f - dz * dz, 0.5f * (0.5f + dz) * (0.5f + dz)};
    for (int n0 = -1; n0 <= 1; ++n0) {
      int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -1; n1 <= 1; ++n1) {
        int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -1; n2 <= 1; ++n2) {
          int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          float W = Wx[n0 + 1] * Wy[n1 + 1] * Wz[n2 + 1];
          atomicAdd(&g_charge_mesh[neighbor012], q * W / para.volume_per_cell);
        }
      }
    }
  }
}

__global__ void find_force(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const cufftComplex* g_mesh_x_real,
  const cufftComplex* g_mesh_y_real,
  const cufftComplex* g_mesh_z_real,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    float q = g_charge[n];
    float sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    float sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    float sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    float reduced_pos[3] = {sx * para.K[0], sy * para.K[1], sz * para.K[2]};
    int ix = int(reduced_pos[0] + 0.5); // can be 0, ..., K[0]
    int iy = int(reduced_pos[1] + 0.5); // can be 0, ..., K[1]
    int iz = int(reduced_pos[2] + 0.5); // can be 0, ..., K[2]
    float dx = reduced_pos[0] - ix; // (-0.5, 0.5)
    float dy = reduced_pos[1] - iy; // (-0.5, 0.5)
    float dz = reduced_pos[2] - iz; // (-0.5, 0.5)
    // Eq. (6.29) in Allen & Tildesley
    float Wx[3] = {0.5f * (0.5f - dx) * (0.5f - dx), 0.75f - dx * dx, 0.5f * (0.5f + dx) * (0.5f + dx)};
    float Wy[3] = {0.5f * (0.5f - dy) * (0.5f - dy), 0.75f - dy * dy, 0.5f * (0.5f + dy) * (0.5f + dy)};
    float Wz[3] = {0.5f * (0.5f - dz) * (0.5f - dz), 0.75f - dz * dz, 0.5f * (0.5f + dz) * (0.5f + dz)};
    float E[3] = {0.0f, 0.0f, 0.0f};
    for (int n0 = -1; n0 <= 1; ++n0) {
      int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -1; n1 <= 1; ++n1) {
        int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -1; n2 <= 1; ++n2) {
          int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          double W = Wx[n0 + 1] * Wy[n1 + 1] * Wz[n2 + 1];
          E[0] += W * g_mesh_x_real[neighbor012].x;
          E[1] += W * g_mesh_y_real[neighbor012].x;
          E[2] += W * g_mesh_z_real[neighbor012].x;
        }
      }
    }
    g_fx[n] = K_C_SP * q * E[0];
    g_fy[n] = K_C_SP * q * E[1];
    g_fz[n] = K_C_SP * q * E[2];
  } 
}

void __global__ find_potential_and_virial(
  const int N,
  const PPPM::Para para,
  const cufftComplex* g_S,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  float* g_virial,
  float* g_pe)
{
  int tid = threadIdx.x;
  int number_of_batches = (para.K0K1K2 - 1) / 1024 + 1;
  __shared__ float s_data[1024];
  float data = 0.0f;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < para.K0K1K2) {
      cufftComplex S = g_S[n];
      float GSS = g_G[n] * (S.x * S.x + S.y * S.y);
      const float kx = g_kx[n];
      const float ky = g_ky[n];
      const float kz = g_kz[n];
      const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / (kx * kx + ky * ky + kz * kz);
      switch (blockIdx.x) {
        case 0:
          data += GSS * (1.0f - alpha_k_factor * kx * kx); // xx
          break;
        case 1:
          data += GSS * (1.0f - alpha_k_factor * ky * ky); // yy
          break;
        case 2:
          data += GSS * (1.0f - alpha_k_factor * kz * kz); // zz
          break;
        case 3:
          data -= GSS * (alpha_k_factor * kx * ky); // xy
          break;
        case 4:
          data -= GSS * (alpha_k_factor * ky * kz); // yz
          break;
        case 5:
          data -= GSS * (alpha_k_factor * kz * kx); // zx
          break;
        case 6:
          data += GSS; // potential
          break;
      }
    }
  }
  s_data[tid] = data;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  number_of_batches = (N - 1) / 1024 + 1;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      switch (blockIdx.x) {
        case 0:
          g_virial[n + 0 * N] += K_C_SP * s_data[0] / N;
          break;
        case 1:
          g_virial[n + 1 * N] += K_C_SP * s_data[0] / N;
          break;
        case 2:
          g_virial[n + 2 * N] += K_C_SP * s_data[0] / N;
          break;
        case 3:
          g_virial[n + 3 * N] += K_C_SP * s_data[0] / N;
          g_virial[n + 6 * N] += K_C_SP * s_data[0] / N;
          break;
        case 4:
          g_virial[n + 5 * N] += K_C_SP * s_data[0] / N;
          g_virial[n + 8 * N] += K_C_SP * s_data[0] / N;
          break;
        case 5:
          g_virial[n + 4 * N] += K_C_SP * s_data[0] / N;
          g_virial[n + 7 * N] += K_C_SP * s_data[0] / N;
          break;
        case 6:
          g_pe[n] += K_C_SP * s_data[0] / N;
          break;
      }
    }
  }
}

}

PPPM::PPPM()
{
  // nothing
}

PPPM::~PPPM()
{
  // nothing
}

void PPPM::initialize(const float alpha_input)
{
  alpha = alpha_input;
  alpha_factor = 0.25f / (alpha * alpha);
  kx.resize(para.K0K1K2);
  ky.resize(para.K0K1K2);
  kz.resize(para.K0K1K2);
  G.resize(para.K0K1K2);
  mesh.resize(para.K0K1K2);
  mesh_fft.resize(para.K0K1K2);
  mesh_fft_x.resize(para.K0K1K2);
  mesh_fft_y.resize(para.K0K1K2);
  mesh_fff_z.resize(para.K0K1K2);
  mesh_fft_x_ifft.resize(para.K0K1K2);
  mesh_fft_y_ifft.resize(para.K0K1K2);
  mesh_fft_z_ifft.resize(para.K0K1K2);
}

void PPPM::find_para(const Box& box)
{
  const float two_pi = 6.2831853f;
  const double mesh_spacing = 1.0; // Is this good enough?
  double volume = box.get_volume();
  for (int d = 0; d < 3; ++d) {
    double box_thickness = volume / box.get_area(d);
    para.K[d] = box_thickness / mesh_spacing;
    para.K[d] = get_best_K(int(para.K[d]));
    para.K_half[d] = para.K[d] / 2;
    para.two_pi_over_K[d] = two_pi / para.K[d];
    std::cout << "K[d]=" << para.K[d] << std::endl;
  }
  para.K0K1 = para.K[0] * para.K[1];
  para.K0K1K2 = para.K0K1 * para.K[2];
  std::cout << "K0K1K2=" << para.K0K1K2 << std::endl;

  float a0[3] = {(float)box.cpu_h[0], (float)box.cpu_h[3], (float)box.cpu_h[6]};
  float a1[3] = {(float)box.cpu_h[1], (float)box.cpu_h[4], (float)box.cpu_h[7]};
  float a2[3] = {(float)box.cpu_h[2], (float)box.cpu_h[5], (float)box.cpu_h[8]};
  float det = a0[0] * (a1[1] * a2[2] - a2[1] * a1[2]) +
              a1[0] * (a2[1] * a0[2] - a0[1] * a2[2]) +
              a2[0] * (a0[1] * a1[2] - a1[1] * a0[2]);
  cross_product(a1, a2, para.b[0]);
  cross_product(a2, a0, para.b[1]);
  cross_product(a0, a1, para.b[2]);
  const float two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] *= two_pi_over_det;
    para.b[1][d] *= two_pi_over_det;
    para.b[2][d] *= two_pi_over_det;
  }
}

void PPPM::find_force(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<float>& D_real,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& potential_per_atom)
{
  find_para(box);
  exit(1);

}
