/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The k-space part of the PPPM method for charge and C6 dispersion.
------------------------------------------------------------------------------*/

#include "pppm.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

namespace{

constexpr int max_mesh_points = 512 * 512 * 512;

bool is_good_K(int n)
{
  const int primes[4] = {2, 3, 5, 7};
  for (int p : primes) {
    while (n % p == 0) {
      n /= p;
    }
  }
  return n == 1;
}

int get_best_K(const double required)
{
  int n = static_cast<int>(std::ceil(std::fmin(required, max_mesh_points)));
  if (n < 16) {
    n = 16;
  }
  if (n % 2 != 0) {
    ++n;
  }
  while (!is_good_K(n)) {
    n += 2;
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
  const PPPM_VDW::Para para,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G_charge,
  float* g_G_vdw,
  float* g_G_vdw_virial)
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
    if (para.has_charge) {
      if (ksq == 0.0f) {
        g_G_charge[n] = 0.0f;
      } else {
        float G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
        G_opt /= denominator[0] * denominator[1] * denominator[2];
        g_G_charge[n] = G_opt;
      }
    }

    // Isele-Holder et al., JCP 137, 174107 (2012), Eqs. (3) and (7).
    // Retain only m=0 in the numerator alias sum, as in the charge channel.
    // G includes the attractive sign and 1/(2V) for a full FFT grid.
    // H = -2 dG/d(k^2), with the mesh factor held fixed under cell strain.
    // Use double here to reduce cancellation in the exponentially small tails.
    const double sqrt_pi = 1.772453850905516;
    const double alpha = para.alpha;
    const double mesh_factor = numerator / double(denominator[0] * denominator[1] * denominator[2]);
    const double prefactor = para.two_pi_over_V * sqrt_pi * mesh_factor;
    const double b2 = double(ksq) * para.alpha_factor;
    const double b = sqrt(b2);
    const double exp_b2 = exp(-b2);
    const double E = exp_b2 - sqrt_pi * b * erfc(b);
    const double B = exp_b2 - 2.0 * b2 * E;
    g_G_vdw[n] = -prefactor * alpha * alpha * alpha * B / 12.0;
    g_G_vdw_virial[n] = ksq == 0.0f ? 0.0f : -prefactor * alpha * E / 8.0;
  }
}

void __global__ set_mesh_to_zero(
  const PPPM_VDW::Para para, gpufftComplex* g_mesh_charge, gpufftComplex* g_mesh_vdw)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    if (para.has_charge) {
      g_mesh_charge[n] = {0.0f, 0.0f};
    }
    g_mesh_vdw[n] = {0.0f, 0.0f};
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
  const PPPM_VDW::Para para,
  const Box box,
  const float* g_charge,
  const float* g_C6,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  gpufftComplex* g_mesh_charge,
  gpufftComplex* g_mesh_vdw)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = para.has_charge ? g_charge[n] : 0.0f;
    const float C6 = g_C6[n];
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
          if (para.has_charge) {
            atomicAdd(&g_mesh_charge[neighbor012].x, q * W);
          }
          atomicAdd(&g_mesh_vdw[neighbor012].x, C6 * W);
        }
      }
    }
  }
}

void __global__ ik_times_mesh_times_G(
  const PPPM_VDW::Para para,
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
  const PPPM_VDW::Para para,
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
  const PPPM_VDW::Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_G_vdw_virial,
  const bool is_vdw,
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
    const float G = g_G[n];
    float H = 0.0f;
    if (is_vdw) {
      H = g_G_vdw_virial[n];
    } else if (ksq != 0.0f) {
      H = G * (2.0f * para.alpha_factor + 2.0f / ksq);
    }
    const gpufftComplex S = g_S[n];
    // Write every mode, including k=0, because the work arrays are reused after IFFT.
    // Dispersion has a nonzero diagonal virial at k=0; charge has G=H=0 there.
    float B = G - H * kx * kx;
    g_mesh_virial_xx[n] = {B * S.x, B * S.y};
    B = G - H * ky * ky;
    g_mesh_virial_yy[n] = {B * S.x, B * S.y};
    B = G - H * kz * kz;
    g_mesh_virial_zz[n] = {B * S.x, B * S.y};
    B = -H * kx * ky;
    g_mesh_virial_xy[n] = {B * S.x, B * S.y};
    B = -H * ky * kz;
    g_mesh_virial_yz[n] = {B * S.x, B * S.y};
    B = -H * kz * kx;
    g_mesh_virial_zx[n] = {B * S.x, B * S.y};
  }
}

__global__ void find_force_from_field(
  const int N1,
  const int N2,
  const PPPM_VDW::Para para,
  const Box box,
  const float prefactor,
  const float* g_source,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const gpufftComplex* g_mesh_G,
  const gpufftComplex* g_mesh_fft_x_ifft,
  const gpufftComplex* g_mesh_fft_y_ifft,
  const gpufftComplex* g_mesh_fft_z_ifft,
  float* g_D,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float source_factor = prefactor * g_source[n] * 2.0f;
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
    float D = 0.0f;
    float E[3] = {0.0f, 0.0f, 0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = get_index_within_mesh(para.K[0], ix + n0);  // can be 0, ..., K[0]-1
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = get_index_within_mesh(para.K[1], iy + n1);  // can be 0, ..., K[1]-1
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = get_index_within_mesh(para.K[2], iz + n2);  // can be 0, ..., K[2]-1
          const int neighbor012 = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D += W * g_mesh_G[neighbor012].x;
          E[0] += W * g_mesh_fft_x_ifft[neighbor012].x;
          E[1] += W * g_mesh_fft_y_ifft[neighbor012].x;
          E[2] += W * g_mesh_fft_z_ifft[neighbor012].x;
        }
      }
    }
    g_D[n] = 2.0f * prefactor * D;
    g_fx[n] += source_factor * E[0];
    g_fy[n] += source_factor * E[1];
    g_fz[n] += source_factor * E[2];
  }
}

__global__ void find_force_virial_potential_from_field(
  const int N,
  const int N1,
  const int N2,
  const PPPM_VDW::Para para,
  const Box box,
  const float prefactor,
  const float* g_source,
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
  float* g_D,
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
    const float source_factor = prefactor * g_source[n];
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
    float D = 0.0f;
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
          D += W * g_mesh_G[neighbor012].x;
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
    g_D[n] = 2.0f * prefactor * D;
    g_fx[n] += 2.0f * source_factor * E[0];
    g_fy[n] += 2.0f * source_factor * E[1];
    g_fz[n] += 2.0f * source_factor * E[2];
    // virial order
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n + 0 * N] += source_factor * V[0]; // xx
    g_virial[n + 1 * N] += source_factor * V[1]; // yy
    g_virial[n + 2 * N] += source_factor * V[2]; // zz
    g_virial[n + 3 * N] += source_factor * V[3]; // xy
    g_virial[n + 6 * N] += source_factor * V[3]; // yx
    g_virial[n + 5 * N] += source_factor * V[4]; // yz
    g_virial[n + 8 * N] += source_factor * V[4]; // zy
    g_virial[n + 4 * N] += source_factor * V[5]; // xz
    g_virial[n + 7 * N] += source_factor * V[5]; // zx
    g_pe[n] += source_factor * D;
  }
}

void __global__ find_potential_and_virial(
  const int N,
  const PPPM_VDW::Para para,
  const gpufftComplex* g_S,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_G_vdw_virial,
  const bool is_vdw,
  const float potential_factor,
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
      const float SS = S.x * S.x + S.y * S.y;
      const float GSS = g_G[n] * SS;
      const float kx = g_kx[n];
      const float ky = g_ky[n];
      const float kz = g_kz[n];
      const float ksq = kx * kx + ky * ky + kz * kz;
      float H = 0.0f;
      if (is_vdw) {
        H = g_G_vdw_virial[n];
      } else if (ksq != 0.0f) {
        H = g_G[n] * (2.0f * para.alpha_factor + 2.0f / ksq);
      }
      const float HSS = H * SS;
      switch (blockIdx.x) {
        case 0:
          data += GSS - HSS * kx * kx; // xx
          break;
        case 1:
          data += GSS - HSS * ky * ky; // yy
          break;
        case 2:
          data += GSS - HSS * kz * kz; // zz
          break;
        case 3:
          data -= HSS * kx * ky; // xy
          break;
        case 4:
          data -= HSS * ky * kz; // yz
          break;
        case 5:
          data -= HSS * kz * kx; // zx
          break;
        case 6:
          data += GSS; // potential, including the dispersion k=0 mode
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
    const int n = tid + batch * 1024;
    if (n < N) {
      // virial order
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      switch (blockIdx.x) {
        case 0:
          g_virial[n + 0 * N] += s_data[0] * potential_factor; // xx
          break;
        case 1:
          g_virial[n + 1 * N] += s_data[0] * potential_factor; // yy
          break;
        case 2:
          g_virial[n + 2 * N] += s_data[0] * potential_factor; // zz
          break;
        case 3:
          g_virial[n + 3 * N] += s_data[0] * potential_factor; // xy
          g_virial[n + 6 * N] += s_data[0] * potential_factor; // yx
          break;
        case 4:
          g_virial[n + 5 * N] += s_data[0] * potential_factor; // yz
          g_virial[n + 8 * N] += s_data[0] * potential_factor; // zy
          break;
        case 5:
          g_virial[n + 4 * N] += s_data[0] * potential_factor; // xz
          g_virial[n + 7 * N] += s_data[0] * potential_factor; // zx
          break;
        case 6:
          g_pe[n] += s_data[0] * potential_factor;
          break;
      }
    }
  }
}

__global__ void find_vdw_self_energy(
  const int N1,
  const int N2,
  const float alpha,
  const float* g_C6,
  float* g_D_C6,
  double* g_pe)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const float alpha_squared = alpha * alpha;
    const float alpha_sixth = alpha_squared * alpha_squared * alpha_squared;
    const float C6 = g_C6[n];
    g_pe[n] += double(alpha_sixth) * C6 * C6 / 12.0;
    g_D_C6[n] += alpha_sixth * C6 / 6.0f;
  }
}

}

PPPM_VDW::PPPM_VDW()
{
  // nothing
}

PPPM_VDW::~PPPM_VDW()
{
  destroy_plans();
}

void PPPM_VDW::destroy_plans()
{
  if (plan_initialized) {
    gpufftDestroy(plan);
    plan_initialized = false;
  }
  if (plan_virial_initialized) {
    gpufftDestroy(plan_virial);
    plan_virial_initialized = false;
  }
}

void PPPM_VDW::allocate_memory()
{
  destroy_plans();
  kernels_initialized = false;
  kx.resize(para.K0K1K2);
  ky.resize(para.K0K1K2);
  kz.resize(para.K0K1K2);
  if (para.has_charge) {
    G_charge.resize(para.K0K1K2);
    mesh_charge.resize(para.K0K1K2);
  }
  G_vdw.resize(para.K0K1K2);
  G_vdw_virial.resize(para.K0K1K2);
  mesh_vdw.resize(para.K0K1K2);
  mesh_G.resize(para.K0K1K2);
  mesh_x.resize(para.K0K1K2);
  mesh_y.resize(para.K0K1K2);
  mesh_z.resize(para.K0K1K2);
  // para.K[2] is the slowest changing dimension; para.K[0] is the fastest changing dimension
  if (gpufftPlan3d(&plan, para.K[2], para.K[1], para.K[0], GPUFFT_C2C) != GPUFFT_SUCCESS) {
    std::cout << "GPUFFT error: Plan creation failed" << std::endl;
    exit(1);
  }
  plan_initialized = true;

  if (need_peratom_virial) {
    mesh_virial.resize(para.K0K1K2 * 6);
    int n[3] = {para.K[2], para.K[1], para.K[0]};
    if (gpufftPlanMany(&plan_virial, 3, n, NULL, 1, para.K0K1K2, NULL, 1, para.K0K1K2, GPUFFT_C2C, 6) != GPUFFT_SUCCESS) {
      std::cout << "GPUFFT error: plan_virial creation failed" << std::endl;
      exit(1);
    }
    plan_virial_initialized = true;
  }
}

void PPPM_VDW::initialize(
  const float alpha_input,
  const bool has_charge_input,
  const bool need_peratom_virial_input,
  const double mesh_spacing_input)
{
  destroy_plans();
  para = {};
  kernels_initialized = false;
  para.has_charge = has_charge_input;
  mesh_spacing = mesh_spacing_input;
  need_peratom_virial = need_peratom_virial_input;
  para.alpha = alpha_input;
  para.alpha_factor = 0.25f / (para.alpha * para.alpha);
}

void PPPM_VDW::find_para(const Box& box)
{
  const float two_pi = 6.2831853f;
  const double volume = box.get_volume();
  para.two_pi_over_V = two_pi / volume;
  for (int d = 0; d < 3; ++d) {
    const double required = volume / box.get_area(d) / mesh_spacing;
    if (required > para.K[d]) {
      para.K[d] = get_best_K(required);
    }
    para.K_half[d] = para.K[d] / 2;
    para.two_pi_over_K[d] = two_pi / para.K[d];
  }
  const double number_of_points = double(para.K[0]) * para.K[1] * para.K[2];
  if (number_of_points > max_mesh_points) {
    PRINT_INPUT_ERROR("PPPM mesh is too large; increase spacing or reduce the box size.");
  }
  if (number_of_points != para.K0K1K2) {
    para.K0K1 = para.K[0] * para.K[1];
    para.K0K1K2 = static_cast<int>(number_of_points);
    allocate_memory();
  }
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * (float)box.cpu_h[9 + d];
    para.b[1][d] = two_pi * (float)box.cpu_h[12 + d];
    para.b[2][d] = two_pi * (float)box.cpu_h[15 + d];
  }
}

void PPPM_VDW::print_mesh(const int N1, const int N2, const Box& box, const GPU_Vector<float>& charge)
{
  const double volume = box.get_volume();
  printf(
    "PPPM %s mesh: %d x %d x %d (target spacing %.17g A; actual spacing %.17g %.17g %.17g A).\n",
    para.has_charge ? "charge-vdW" : "vdW",
    para.K[0],
    para.K[1],
    para.K[2],
    mesh_spacing,
    volume / box.get_area(0) / para.K[0],
    volume / box.get_area(1) / para.K[1],
    volume / box.get_area(2) / para.K[2]);

  if (!para.has_charge) {
    return;
  }

  // This estimate applies only to the charge channel.
  // Root-mean-square force error for ik differentiation and five-point charge assignment
  // with the optimal influence function, Eq. (38) of M. Deserno and C. Holm, JCP 109, 7694 (1998).
  // The coefficients are a_m^(5) of Table II therein.
  // Eq. (38) holds for a cubic box. Here it is evaluated per direction with the thickness of the
  // cell as L, and the root mean square of the three values is reported.
  const double coefficients[5] = {
    1.0 / 23232.0,
    7601.0 / 13628160.0,
    143.0 / 69120.0,
    517231.0 / 106536960.0,
    106640677.0 / 11737571328.0};
  const int number_of_charges = N2 - N1;
  std::vector<float> q(number_of_charges);
  CHECK(gpuMemcpy(
    q.data(), charge.data() + N1, sizeof(float) * number_of_charges, gpuMemcpyDeviceToHost));
  double sum_of_squared_charges = 0.0;
  for (const float q_i : q) {
    sum_of_squared_charges += double(q_i) * q_i;
  }
  const double alpha = para.alpha;
  double squared_error = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double thickness = volume / box.get_area(d);
    const double h_alpha = thickness / para.K[d] * alpha;
    double sum = 0.0;
    for (int m = 0; m < 5; ++m) {
      sum += coefficients[m] * std::pow(h_alpha, 2 * m);
    }
    const double error =
      K_C_SP * sum_of_squared_charges * std::pow(h_alpha, 5) *
      std::sqrt(alpha * thickness * std::sqrt(2.0 * PI) * sum / number_of_charges) /
      (thickness * thickness);
    squared_error += error * error;
  }
  printf("PPPM charge force error estimate: %.3e eV/A.\n", std::sqrt(squared_error / 3.0));
}

void PPPM_VDW::find_force(
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<float>& C6,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<float>& D_real,
  GPU_Vector<float>& D_C6,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& potential_per_atom)
{
  const bool first_mesh = !plan_initialized;
  find_para(box);
  if (first_mesh) {
    print_mesh(N1, N2, box, charge);
  }

  // The influence functions depend on the box, grid and fixed alpha, not on the sources.
  if (!kernels_initialized || !std::equal(box.cpu_h, box.cpu_h + 9, kernel_box)) {
    find_k_and_G_opt<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
      para,
      kx.data(),
      ky.data(),
      kz.data(),
      para.has_charge ? G_charge.data() : nullptr,
      G_vdw.data(),
      G_vdw_virial.data());
    GPU_CHECK_KERNEL
    std::copy(box.cpu_h, box.cpu_h + 9, kernel_box);
    kernels_initialized = true;
  }

  set_mesh_to_zero<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para,
    para.has_charge ? mesh_charge.data() : nullptr,
    mesh_vdw.data());
  GPU_CHECK_KERNEL

  // Both sources use the same grid indices and five-point interpolation weights.
  find_mesh<<<(N2 - N1 - 1) / 64 + 1, 64>>>(
    N1,
    N2,
    para,
    box,
    para.has_charge ? charge.data() : nullptr,
    C6.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    para.has_charge ? mesh_charge.data() : nullptr,
    mesh_vdw.data());
  GPU_CHECK_KERNEL

  if (para.has_charge) {
    find_force_channel(
      false, N, N1, N2, box, charge, position_per_atom, D_real,
      force_per_atom, virial_per_atom, potential_per_atom);
  }
  find_force_channel(
    true, N, N1, N2, box, C6, position_per_atom, D_C6,
    force_per_atom, virial_per_atom, potential_per_atom);

  find_vdw_self_energy<<<(N2 - N1 - 1) / 64 + 1, 64>>>(
    N1, N2, para.alpha, C6.data(), D_C6.data(), potential_per_atom.data());
  GPU_CHECK_KERNEL
}

void PPPM_VDW::find_force_channel(
  const bool is_vdw,
  const int N,
  const int N1,
  const int N2,
  const Box& box,
  const GPU_Vector<float>& source,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<float>& D,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& potential_per_atom)
{
  GPU_Vector<gpufftComplex>& mesh = is_vdw ? mesh_vdw : mesh_charge;
  const GPU_Vector<float>& G = is_vdw ? G_vdw : G_charge;
  const float* G_virial = is_vdw ? G_vdw_virial.data() : nullptr;
  const float prefactor = is_vdw ? 1.0f : K_C_SP;

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
      G_virial,
      is_vdw,
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
      prefactor,
      source.data(),
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
      D.data(),
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
      prefactor,
      source.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      mesh_G.data(),
      mesh_x.data(),
      mesh_y.data(),
      mesh_z.data(),
      D.data(),
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
      G_virial,
      is_vdw,
      prefactor / N,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}
