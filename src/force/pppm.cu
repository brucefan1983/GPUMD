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

    // Eq. (6.40) in Allen & Tildesley
    float denominator[3] = {0.0f};
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) {
        nk[d] -= para.K[d];
      }
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
    if (ksq == 0.0f) {
      g_G[n] = 0.0f;
    } else {
      float G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
      G_opt /= denominator[0] * denominator[1] * denominator[2];
      g_G[n] = G_opt;
    }
  }
}

void __global__ set_charge_mesh_to_zero(const PPPM::Para para, cufftComplex* g_charge_mesh)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    g_charge_mesh[n].x = 0.0f;
    g_charge_mesh[n].y = 0.0f;
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

__global__ void find_charge_mesh(
  const int N1,
  const int N2,
  const PPPM::Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  cufftComplex* g_charge_mesh)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    float q = g_charge[n];
    float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    int ix = int(sx + 0.5); // can be 0, ..., K[0]
    int iy = int(sy + 0.5); // can be 0, ..., K[1]
    int iz = int(sz + 0.5); // can be 0, ..., K[2]
    float dx = sx - ix; // (-0.5, 0.5)
    float dy = sy - iy; // (-0.5, 0.5)
    float dz = sz - iz; // (-0.5, 0.5)
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
          atomicAdd(&g_charge_mesh[neighbor012].x, q * W);
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
  const cufftComplex* g_mesh_fft,
  cufftComplex* g_mesh_fft_x,
  cufftComplex* g_mesh_fft_y,
  cufftComplex* g_mesh_fft_z)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    float kx = g_kx[n];
    float ky = g_ky[n];
    float kz = g_kz[n];
    float G = g_G[n];
    cufftComplex mesh_fft = g_mesh_fft[n];
    g_mesh_fft_x[n] = {mesh_fft.y * kx * G, -mesh_fft.x * kx * G};
    g_mesh_fft_y[n] = {mesh_fft.y * ky * G, -mesh_fft.x * ky * G};
    g_mesh_fft_z[n] = {mesh_fft.y * kz * G, -mesh_fft.x * kz * G};
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
  const cufftComplex* g_mesh_fft_x_ifft,
  const cufftComplex* g_mesh_fft_y_ifft,
  const cufftComplex* g_mesh_fft_z_ifft,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    double x = g_x[n];
    double y = g_y[n];
    double z = g_z[n];
    float q = K_C_SP * g_charge[n] * 2.0f;
    float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    int ix = int(sx + 0.5); // can be 0, ..., K[0]
    int iy = int(sy + 0.5); // can be 0, ..., K[1]
    int iz = int(sz + 0.5); // can be 0, ..., K[2]
    float dx = sx - ix; // (-0.5, 0.5)
    float dy = sy - iy; // (-0.5, 0.5)
    float dz = sz - iz; // (-0.5, 0.5)
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
          float W = Wx[n0 + 1] * Wy[n1 + 1] * Wz[n2 + 1];
          E[0] += W * g_mesh_fft_x_ifft[neighbor012].x;
          E[1] += W * g_mesh_fft_y_ifft[neighbor012].x;
          E[2] += W * g_mesh_fft_z_ifft[neighbor012].x;
        }
      }
    }
    g_fx[n] = q * E[0];
    g_fy[n] = q * E[1];
    g_fz[n] = q * E[2];
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
  double* g_virial,
  double* g_pe)
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
    int n = tid + batch * 1024;
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
  // nothing
}

void PPPM::allocate_memory()
{
  kx.resize(para.K0K1K2);
  ky.resize(para.K0K1K2);
  kz.resize(para.K0K1K2);
  G.resize(para.K0K1K2);
  mesh.resize(para.K0K1K2);
  mesh_fft.resize(para.K0K1K2);
  mesh_fft_x.resize(para.K0K1K2);
  mesh_fft_y.resize(para.K0K1K2);
  mesh_fft_z.resize(para.K0K1K2);
  mesh_fft_x_ifft.resize(para.K0K1K2);
  mesh_fft_y_ifft.resize(para.K0K1K2);
  mesh_fft_z_ifft.resize(para.K0K1K2);
}

void PPPM::initialize(const float alpha_input)
{
  para.alpha = alpha_input;
  para.alpha_factor = 0.25f / (para.alpha * para.alpha);
  allocate_memory();
}

void PPPM::find_para(const int N, const Box& box)
{
  const float two_pi = 6.2831853f;
  const double mesh_spacing = 1.0; // Is this good enough?
  double volume = box.get_volume();
  para.two_pi_over_V = two_pi / volume;
  for (int d = 0; d < 3; ++d) {
    double box_thickness = volume / box.get_area(d);
    para.K[d] = box_thickness / mesh_spacing;
    para.K[d] = get_best_K(para.K[d]);
    para.K_half[d] = para.K[d] / 2;
    para.two_pi_over_K[d] = two_pi / para.K[d];
    std::cout << "K[d]=" << para.K[d] << std::endl;
  }
  para.K0K1 = para.K[0] * para.K[1];
  int K0K1K2 = para.K0K1 * para.K[2];
  if (K0K1K2 > para.K0K1K2) {
    std::cout << "old K0K1K2=" << para.K0K1K2 << std::endl;
    para.K0K1K2 = K0K1K2;
    std::cout << "new K0K1K2=" << para.K0K1K2 << std::endl;
    allocate_memory();
  }
  para.potential_factor = K_C_SP / N;

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
  std::cout << "b[0]=" << para.b[0][0] << " " << para.b[0][1] << " " << para.b[0][2] << std::endl;
  std::cout << "b[1]=" << para.b[1][0] << " " << para.b[1][1] << " " << para.b[1][2] << std::endl;
  std::cout << "b[2]=" << para.b[2][0] << " " << para.b[2][1] << " " << para.b[2][2] << std::endl;
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

  set_charge_mesh_to_zero<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(para, mesh.data());
  GPU_CHECK_KERNEL

  find_charge_mesh<<<(N - 1) / 64 + 1, 64>>>(
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

  cufftHandle plan; // optimize later

  // para.K[2] is the slowest changing dimension; para.K[0] is the fastest changing dimension
  if (cufftPlan3d(&plan, para.K[2], para.K[1], para.K[0], CUFFT_C2C) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: Plan creation failed" << std::endl;
    exit(1);
  }

  if (cufftExecC2C(plan, mesh.data(), mesh_fft.data(), CUFFT_FORWARD) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: ExecC2C Forward failed" << std::endl;
    exit(1);
  }

  ik_times_mesh_times_G<<<(para.K0K1K2 - 1) / 64 + 1, 64>>>(
    para,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    mesh_fft.data(),
    mesh_fft_x.data(),
    mesh_fft_y.data(),
    mesh_fft_z.data());
  GPU_CHECK_KERNEL

  if (cufftExecC2C(plan, mesh_fft_x.data(), mesh_fft_x_ifft.data(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (cufftExecC2C(plan, mesh_fft_y.data(), mesh_fft_y_ifft.data(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  if (cufftExecC2C(plan, mesh_fft_z.data(), mesh_fft_z_ifft.data(), CUFFT_INVERSE) != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: ExecC2C Inverse failed" << std::endl;
    exit(1);
  }

  cufftDestroy(plan); // optimize later

  find_force_from_field<<<(N - 1) / 64 + 1, 64>>>(
    N1,
    N2,
    para,
    box,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    mesh_fft_x_ifft.data(),
    mesh_fft_y_ifft.data(),
    mesh_fft_z_ifft.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2);
  GPU_CHECK_KERNEL

  find_potential_and_virial<<<7, 1024>>>(
    N,
    para,
    mesh_fft.data(),
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    virial_per_atom.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
}
