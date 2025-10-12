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
#include "utilities/read_file.cuh"
#include <cmath>
#include <vector>
#include <iostream>

namespace{

int get_best_K(const int m)
{
  int n = 16;
  while (n < m) {
    n *= 2;
  }
  return n;
}

__constant__ float sinc_coeff[6] = {1.0f, -1.6666667e-1f, 8.3333333e-3f, -1.9841270e-4f, 2.7557319e-6f, -2.5052108e-8f};
__constant__ float G_coeff[5] = {1.0000000e+00f, -1.6666667e+00f, 7.7777778e-01f, -8.9947090e-02f, 7.0546737e-04f};
__constant__ float W_coeff[5][5] = {
  {2.6041667e-03f, -2.0833333e-02f, 6.2500000e-02f, -8.3333333e-02f, 4.1666667e-02f},
  {1.9791667e-01f, -4.5833333e-01f, 2.5000000e-01f, 1.6666667e-01f, -1.6666667e-01f},
  {5.9895833e-01f, 0.0000000e+00f, -6.2500000e-01f, 0.0000000e+00f, 2.5000000e-01f},
  {1.9791667e-01f, 4.5833333e-01f, 2.5000000e-01f, -1.6666667e-01f, -1.6666667e-01f},
  {2.6041667e-03f, 2.0833333e-02f, 6.2500000e-02f, 8.3333333e-02f, 4.1666667e-02f}
};

__device__ inline float sinc(const float x)
{
  float y = 0.0f;
  if (x * x <= 1.0f) {
    float term = 1.0f;
    for (int i = 0; i < 6; ++i) {
      y += sinc_coeff[i] * term;
      term *= x * x;
    }
  } else {
    y = sin(x) / x;
  }
  return y;
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

    // Eqs. (2.25) and (2.26) in V. Ballenegger, J. J. Cerda, and C. Holm, JCTC 8, 936 (2012)
    float denominator[3] = {0.0f};
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) {
        nk[d] -= para.K[d];
      }
      float t = sin(0.5f * para.two_pi_over_K[d] * nk[d]);
      t *= t;
      t = (((G_coeff[4] * t + G_coeff[3]) * t + G_coeff[2]) * t + G_coeff[1]) * t + G_coeff[0];
      denominator[d] = t * t;
    }
    const float kx = nk[0] * para.b[0][0] + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    const float ky = nk[0] * para.b[0][1] + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    const float kz = nk[0] * para.b[0][2] + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    g_kx[n] = kx;
    g_ky[n] = ky;
    g_kz[n] = kz;
    const float ksq = kx * kx + ky * ky + kz * kz;

    // Eqs. (2.21) and (2.25) in V. Ballenegger, J. J. Cerda, and C. Holm, JCTC 8, 936 (2012)
    float numerator = sinc(0.5f * para.two_pi_over_K[0] * nk[0]);
    numerator *= sinc(0.5f * para.two_pi_over_K[1] * nk[1]);
    numerator *= sinc(0.5f * para.two_pi_over_K[2] * nk[2]);
    numerator = numerator * numerator * numerator * numerator * numerator;
    numerator *= numerator;

    // Eqs. (2.25) in V. Ballenegger, J. J. Cerda, and C. Holm, JCTC 8, 936 (2012)
    if (ksq == 0.0f) {
      g_G[n] = 0.0f;
    } else {
      float G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
      G_opt /= denominator[0] * denominator[1] * denominator[2];
      g_G[n] = G_opt;
    }
  }
}

void __global__ set_mesh_to_zero(const PPPM::Para para, gpufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    g_mesh[n].x = 0.0f;
    g_mesh[n].y = 0.0f;
  }
}

__device__ inline int get_index_within_mesh(const int K, const int n)
{
  int y = n;
  if (n >= K) {
    y = n - K;
  } else if (n < 0) {
    y = n + K;
  }
  return y;
}

__global__ void find_mesh(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  gpufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = g_charge[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f); // can be 0, ..., K[0]
    const int iy = int(sy + 0.5f); // can be 0, ..., K[1]
    const int iz = int(sz + 0.5f); // can be 0, ..., K[2]
    const float dx = sx - ix; // (-0.5, 0.5)
    const float dy = sy - iy; // (-0.5, 0.5)
    const float dz = sz - iz; // (-0.5, 0.5)
    // Appendix E in M. Deserno and C. Holm, JCP 109, 7678 (1998)
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          atomicAdd(&g_mesh[neighbor012].x, q * W);
        }
      }
    }
  }
}

void __global__ ik_times_mesh_times_G(
  const PPPM::Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const gpufftComplex* g_mesh_fft,
  gpufftComplex* g_mesh_fft_x,
  gpufftComplex* g_mesh_fft_y,
  gpufftComplex* g_mesh_fft_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float G = g_G[n];
    gpufftComplex mesh_fft = g_mesh_fft[n];
    g_mesh_fft_x[n] = {mesh_fft.y * kx * G, -mesh_fft.x * kx * G};
    g_mesh_fft_y[n] = {mesh_fft.y * ky * G, -mesh_fft.x * ky * G};
    g_mesh_fft_z[n] = {mesh_fft.y * kz * G, -mesh_fft.x * kz * G};
  }
}

void __global__ find_mesh_G(
  const PPPM::Para para,
  const float* g_G,
  const gpufftComplex* g_mesh,
  gpufftComplex* g_mesh_G)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float G = g_G[n];
    gpufftComplex mesh = g_mesh[n];
    g_mesh_G[n] = {mesh.x * G, mesh.y * G};
  }
}

void __global__ find_mesh_virial(
  const PPPM::Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const gpufftComplex* g_S,
  gpufftComplex* g_mesh_virial_xx,
  gpufftComplex* g_mesh_virial_yy,
  gpufftComplex* g_mesh_virial_zz,
  gpufftComplex* g_mesh_virial_xy,
  gpufftComplex* g_mesh_virial_yz,
  gpufftComplex* g_mesh_virial_zx)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float ksq = kx * kx + ky * ky + kz * kz;
    if (ksq != 0.0f) {
      const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
      const float G = g_G[n];
      const gpufftComplex S = g_S[n];
      const float GSx = G * S.x;
      const float GSy = G * S.y;
      float B = 1.0f - alpha_k_factor * kx * kx;
      g_mesh_virial_xx[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * ky * ky;
      g_mesh_virial_yy[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * kz * kz;
      g_mesh_virial_zz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kx * ky;
      g_mesh_virial_xy[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * ky * kz;
      g_mesh_virial_yz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kz * kx;
      g_mesh_virial_zx[n] = {B * GSx, B * GSy};
    }
  }
}

__global__ void find_force_from_field(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = K_C_SP * g_charge[n] * 2.0f;
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f); // can be 0, ..., K[0]
    const int iy = int(sy + 0.5f); // can be 0, ..., K[1]
    const int iz = int(sz + 0.5f); // can be 0, ..., K[2]
    const float dx = sx - ix; // (-0.5, 0.5)
    const float dy = sy - iy; // (-0.5, 0.5)
    const float dz = sz - iz; // (-0.5, 0.5)
    // Appendix E in M. Deserno and C. Holm, JCP 109, 7678 (1998)
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    float D_real = 0.0f;
    float E[3] = {0.0f, 0.0f, 0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D_real += W * g_mesh_G[neighbor012].x;
          E[0] += W * g_mesh_fft_x_ifft[neighbor012].x;
          E[1] += W * g_mesh_fft_y_ifft[neighbor012].x;
          E[2] += W * g_mesh_fft_z_ifft[neighbor012].x;
        }
      }
    }
    g_D_real[n] = 2.0f * K_C_SP * D_real;
    g_fx[n] += q * E[0];
    g_fy[n] += q * E[1];
    g_fz[n] += q * E[2];
  } 
}

__global__ void find_force_virial_potential_from_field(
  const int N,
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  const gpufftComplex* g_mesh_virial_xx,
  const gpufftComplex* g_mesh_virial_yy,
  const gpufftComplex* g_mesh_virial_zz,
  const gpufftComplex* g_mesh_virial_xy,
  const gpufftComplex* g_mesh_virial_yz,
  const gpufftComplex* g_mesh_virial_zx,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = K_C_SP * g_charge[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f); // can be 0, ..., K[0]
    const int iy = int(sy + 0.5f); // can be 0, ..., K[1]
    const int iz = int(sz + 0.5f); // can be 0, ..., K[2]
    const float dx = sx - ix; // (-0.5, 0.5)
    const float dy = sy - iy; // (-0.5, 0.5)
    const float dz = sz - iz; // (-0.5, 0.5)
    // Appendix E in M. Deserno and C. Holm, JCP 109, 7678 (1998)
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((W_coeff[d][4] * dx + W_coeff[d][3]) * dx + W_coeff[d][2]) * dx + W_coeff[d][1]) * dx + W_coeff[d][0];
      Wy[d] = (((W_coeff[d][4] * dy + W_coeff[d][3]) * dy + W_coeff[d][2]) * dy + W_coeff[d][1]) * dy + W_coeff[d][0];
      Wz[d] = (((W_coeff[d][4] * dz + W_coeff[d][3]) * dz + W_coeff[d][2]) * dz + W_coeff[d][1]) * dz + W_coeff[d][0];
    }
    float D_real = 0.0f;
    float E[3] = {0.0f, 0.0f, 0.0f};
    float V[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D_real += W * g_mesh_G[neighbor012].x;
          E[0] += W * g_mesh_fft_x_ifft[neighbor012].x;
          E[1] += W * g_mesh_fft_y_ifft[neighbor012].x;
          E[2] += W * g_mesh_fft_z_ifft[neighbor012].x;
          V[0] += W * g_mesh_virial_xx[neighbor012].x;
          V[1] += W * g_mesh_virial_yy[neighbor012].x;
          V[2] += W * g_mesh_virial_zz[neighbor012].x;
          V[3] += W * g_mesh_virial_xy[neighbor012].x;
          V[4] += W * g_mesh_virial_yz[neighbor012].x;
          V[5] += W * g_mesh_virial_zx[neighbor012].x;
        }
      }
    }
    g_D_real[n] = 2.0f * K_C_SP * D_real;
    g_fx[n] += 2.0f * q * E[0];
    g_fy[n] += 2.0f * q * E[1];
    g_fz[n] += 2.0f * q * E[2];
    // virial order
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n + 0 * N] += q * V[0]; // xx
    g_virial[n + 1 * N] += q * V[1]; // yy
    g_virial[n + 2 * N] += q * V[2]; // zz
    g_virial[n + 3 * N] += q * V[3]; // xy
    g_virial[n + 6 * N] += q * V[3]; // yx
    g_virial[n + 5 * N] += q * V[4]; // yz
    g_virial[n + 8 * N] += q * V[4]; // zy
    g_virial[n + 4 * N] += q * V[5]; // xz
    g_virial[n + 7 * N] += q * V[5]; // zx
    g_pe[n] += q * D_real;
  } 
}

void __global__ find_potential_and_virial(
  const int N,
  const PPPM::Para para,
  const gpufftComplex* g_S,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  double* g_virial,
  double* g_pe)
{
  const int tid = threadIdx.x;
  int number_of_batches = (para.K0K1K2 - 1) / 1024 + 1;
  __shared__ float s_data[1024];
  float data = 0.0f;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < para.K0K1K2) {
      gpufftComplex S = g_S[n];
      const float GSS = g_G[n] * (S.x * S.x + S.y * S.y);
      const float kx = g_kx[n];
      const float ky = g_ky[n];
      const float kz = g_kz[n];
      const float ksq = kx * kx + ky * ky + kz * kz;
      if (ksq != 0.0f) {
        const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
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
    const int n = tid + batch * 1024;
    if (n < N) {
      // virial order
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      switch (blockIdx.x) {
        case 0:
          g_virial[n + 0 * N] += s_data[0] * para.potential_factor; // xx
          break;
        case 1:
          g_virial[n + 1 * N] += s_data[0] * para.potential_factor; // yy
          break;
        case 2:
          g_virial[n + 2 * N] += s_data[0] * para.potential_factor; // zz
          break;
        case 3:
          g_virial[n + 3 * N] += s_data[0] * para.potential_factor; // xy
          g_virial[n + 6 * N] += s_data[0] * para.potential_factor; // yx
          break;
        case 4:
          g_virial[n + 5 * N] += s_data[0] * para.potential_factor; // yz
          g_virial[n + 8 * N] += s_data[0] * para.potential_factor; // zy
          break;
        case 5:
          g_virial[n + 4 * N] += s_data[0] * para.potential_factor; // xz
          g_virial[n + 7 * N] += s_data[0] * para.potential_factor; // zx
          break;
        case 6:
          g_pe[n] += s_data[0] * para.potential_factor;
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
  gpufftDestroy(plan);
  if (need_peratom_virial) {
    gpufftDestroy(plan_virial);
  }
}

void PPPM::allocate_memory()
{
  kx.resize(para.K0K1K2);
  ky.resize(para.K0K1K2);
  kz.resize(para.K0K1K2);
  G.resize(para.K0K1K2);
  mesh.resize(para.K0K1K2);
  mesh_G.resize(para.K0K1K2);
  mesh_x.resize(para.K0K1K2);
  mesh_y.resize(para.K0K1K2);
  mesh_z.resize(para.K0K1K2);
  // para.K[2] is the slowest changing dimension; para.K[0] is the fastest changing dimension
  if (gpufftPlan3d(&plan, para.K[2], para.K[1], para.K[0], GPUFFT_C2C) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: Plan creation failed" << std::endl;
    exit(1);
  }

  if (need_peratom_virial) {
    mesh_virial.resize(para.K0K1K2 * 6);
    int n[3] = {para.K[2], para.K[1], para.K[0]};
    if (gpufftPlanMany(&plan_virial, 3, n, NULL, 1, para.K0K1K2, NULL, 1, para.K0K1K2, GPUFFT_C2C, 6) != GPUFFT_SUCCESS) {
      std::cout << "GPUFFT error: plan_virial creation failed" << std::endl;
      exit(1);
    }
  }
}

void PPPM::initialize(const float alpha_input)
{
  need_peratom_virial = check_need_peratom_virial();
  para.alpha = alpha_input;
  para.alpha_factor = 0.25f / (para.alpha * para.alpha);
  para.K[0] = 16;
  para.K[1] = 16;
  para.K[2] = 16;
  para.K0K1K2 = para.K[0] * para.K[1] * para.K[2];
  allocate_memory();
}

void PPPM::find_para(const int N, const Box& box)
{
  const float two_pi = 6.2831853f;
  const double mesh_spacing = 1.0; // Is this good enough?
  const double volume = box.get_volume();
  para.two_pi_over_V = two_pi / volume;
  int K[3] = {0};
  for (int d = 0; d < 3; ++d) {
    const double box_thickness = volume / box.get_area(d);
    K[d] = box_thickness / mesh_spacing;
    K[d] = get_best_K(K[d]);
    para.K_half[d] = K[d] / 2;
    para.two_pi_over_K[d] = two_pi / K[d];
  }
  para.K0K1 = K[0] * K[1];
  para.K0K1K2 = para.K0K1 * K[2];
  if (K[0] != para.K[0] || K[1] != para.K[1] || K[2] != para.K[2]) {
    para.K[0] = K[0];
    para.K[1] = K[1];
    para.K[2] = K[2];
    allocate_memory();
  }
  para.potential_factor = K_C_SP / N;
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * (float)box.cpu_h[9 + d];
    para.b[1][d] = two_pi * (float)box.cpu_h[12 + d];
    para.b[2][d] = two_pi * (float)box.cpu_h[15 + d];
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
  find_para(N, box);

  find_k_and_G_opt<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para, 
    kx.data(), 
    ky.data(), 
    kz.data(), 
    G.data());
  GPU_CHECK_KERNEL

  set_mesh_to_zero<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(para, mesh.data());
  GPU_CHECK_KERNEL

  find_mesh<<<(N - 1) / 64 + 1, 64>>>(
    N1,
    N2,
    para,
    box,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    mesh.data());
  GPU_CHECK_KERNEL

  if (gpufftExecC2C(plan, mesh.data(), mesh.data(), GPUFFT_FORWARD) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Forward failed" << std::endl;
    exit(1);
  }

  ik_times_mesh_times_G<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    mesh.data(),
    mesh_x.data(),
    mesh_y.data(),
    mesh_z.data());
  GPU_CHECK_KERNEL

  find_mesh_G<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para,
    G.data(),
    mesh.data(),
    mesh_G.data());
  GPU_CHECK_KERNEL

  if (need_peratom_virial) {
    find_mesh_virial<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
      para,
      kx.data(),
      ky.data(),
      kz.data(),
      G.data(),
      mesh.data(),
      mesh_virial.data() + para.K0K1K2 * 0,
      mesh_virial.data() + para.K0K1K2 * 1,
      mesh_virial.data() + para.K0K1K2 * 2,
      mesh_virial.data() + para.K0K1K2 * 3,
      mesh_virial.data() + para.K0K1K2 * 4,
      mesh_virial.data() + para.K0K1K2 * 5);
    GPU_CHECK_KERNEL
  }

  if (gpufftExecC2C(plan, mesh_G.data(), mesh_G.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(plan, mesh_x.data(), mesh_x.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(plan, mesh_y.data(), mesh_y.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (gpufftExecC2C(plan, mesh_z.data(), mesh_z.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (need_peratom_virial) {
    if (gpufftExecC2C(plan_virial, mesh_virial.data(), mesh_virial.data(), GPUFFT_INVERSE) != GPUFFT_SUCCESS) {
      std::cout << "GPUFFT error: ExecC2C Inverse failed" << std::endl;
      exit(1);
    }

    // get force, virial, and potential in single kernel
    find_force_virial_potential_from_field<<<(N - 1) / 64 + 1, 64>>>(
      N,
      N1,
      N2,
      para,
      box,
      charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      mesh_G.data(),
      mesh_x.data(),
      mesh_y.data(),
      mesh_z.data(),
      mesh_virial.data() + para.K0K1K2 * 0,
      mesh_virial.data() + para.K0K1K2 * 1,
      mesh_virial.data() + para.K0K1K2 * 2,
      mesh_virial.data() + para.K0K1K2 * 3,
      mesh_virial.data() + para.K0K1K2 * 4,
      mesh_virial.data() + para.K0K1K2 * 5,
      D_real.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  } else {
    // get force only
    find_force_from_field<<<(N - 1) / 64 + 1, 64>>>(
      N1,
      N2,
      para,
      box,
      charge.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      mesh_G.data(),
      mesh_x.data(),
      mesh_y.data(),
      mesh_z.data(),
      D_real.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2);
    GPU_CHECK_KERNEL

    // then get average potential and virial
    find_potential_and_virial<<<7, 1024>>>(
      N,
      para,
      mesh.data(),
      kx.data(),
      ky.data(),
      kz.data(),
      G.data(),
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}
