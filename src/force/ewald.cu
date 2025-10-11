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
The k-space part of the Ewald summation
------------------------------------------------------------------------------*/

#include "ewald.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <cmath>
#include <vector>

Ewald::Ewald()
{
  // nothing
}

Ewald::~Ewald()
{
  // nothing
}

void Ewald::initialize(const float alpha_input)
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
  alpha = alpha_input;
  alpha_factor = 0.25f / (alpha * alpha);
  kx.resize(num_kpoints_max);
  ky.resize(num_kpoints_max);
  kz.resize(num_kpoints_max);
  G.resize(num_kpoints_max);
  S_real.resize(num_kpoints_max);
  S_imag.resize(num_kpoints_max);
}

static void cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] =  a[1] * b [2] - a[2] * b [1];
  c[1] =  a[2] * b [0] - a[0] * b [2];
  c[2] =  a[0] * b [1] - a[1] * b [0];
}

static float get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

void Ewald::find_k_and_G(const double* box)
{
  float a1[3] = {0.0f};
  float a2[3] = {0.0f};
  float a3[3] = {0.0f};
  float det = box[0] * (box[4] * box[8] - box[5] * box[7]) +
    box[1] * (box[5] * box[6] - box[3] * box[8]) +
    box[2] * (box[3] * box[7] - box[4] * box[6]);
  a1[0] = box[0];
  a1[1] = box[3];
  a1[2] = box[6];
  a2[0] = box[1];
  a2[1] = box[4];
  a2[2] = box[7];
  a3[0] = box[2];
  a3[1] = box[5];
  a3[2] = box[8];
  float b1[3] = {0.0f};
  float b2[3] = {0.0f};
  float b3[3] = {0.0f};
  cross_product(a2, a3, b1);
  cross_product(a3, a1, b2);
  cross_product(a1, a2, b3);

  const float two_pi = 6.2831853f;
  const float two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    b1[d] *= two_pi_over_det;
    b2[d] *= two_pi_over_det;
    b3[d] *= two_pi_over_det;
  }

  const float volume_k = two_pi * two_pi * two_pi / abs(det);
  int n1_max = alpha * two_pi * get_area(b2, b3) / volume_k;
  int n2_max = alpha * two_pi * get_area(b3, b1) / volume_k;
  int n3_max = alpha * two_pi * get_area(b1, b2) / volume_k;
  float ksq_max = two_pi * two_pi * alpha * alpha;

  std::vector<float> cpu_kx;
  std::vector<float> cpu_ky;
  std::vector<float> cpu_kz;
  std::vector<float> cpu_G;

#ifdef USE_RBE
  std::uniform_real_distribution<float> rand_number(0.0f, 1.0f);
  float normalization_factor = 0.0f;
  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = - n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = - n3_max; n3 <= n3_max; ++n3) {
        const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
        if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0)) continue;
        const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
        const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
        const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
        const float ksq = kx * kx + ky * ky + kz * kz;
        if (ksq < ksq_max) {
          float exp_factor = exp(-ksq * alpha_factor);
          normalization_factor += exp_factor;
          if (rand_number(rng) < exp_factor) {
            cpu_kx.emplace_back(kx);
            cpu_ky.emplace_back(ky);
            cpu_kz.emplace_back(kz);
            float G = abs(two_pi_over_det) / ksq;
            cpu_G.emplace_back(2.0f * G);
          }
        }
      }
    }
  }

  int num_kpoints = int(cpu_kx.size());
  for (int n = 0; n < num_kpoints; ++n) {
    cpu_G[n] *= normalization_factor / num_kpoints;
  }

#else

  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = - n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = - n3_max; n3 <= n3_max; ++n3) {
        const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
        if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0)) continue;
        const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
        const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
        const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
        const float ksq = kx * kx + ky * ky + kz * kz;
        if (ksq < ksq_max) {
          cpu_kx.emplace_back(kx);
          cpu_ky.emplace_back(ky);
          cpu_kz.emplace_back(kz);
          const float G = abs(two_pi_over_det) / ksq * exp(-ksq * alpha_factor);
          cpu_G.emplace_back(2.0f * G);
        }
      }
    }
  }

  int num_kpoints = int(cpu_kx.size());
#endif

  if (num_kpoints > num_kpoints_max) {
    num_kpoints_max = num_kpoints;
    kx.resize(num_kpoints_max);
    ky.resize(num_kpoints_max);
    kz.resize(num_kpoints_max);
    G.resize(num_kpoints_max);
    S_real.resize(num_kpoints_max);
    S_imag.resize(num_kpoints_max);
  }

  kx.copy_from_host(cpu_kx.data(), num_kpoints);
  ky.copy_from_host(cpu_ky.data(), num_kpoints);
  kz.copy_from_host(cpu_kz.data(), num_kpoints);
  G.copy_from_host(cpu_G.data(), num_kpoints);
}

static __global__ void find_structure_factor(
  const int num_kpoints_max,
  const int N1,
  const int N2,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  float* g_S_real,
  float* g_S_imag)
{
  int nk = blockIdx.x * blockDim.x + threadIdx.x;
  if (nk < num_kpoints_max) {
    float S_real = 0.0f;
    float S_imag = 0.0f;
    for (int n = N1; n < N2; ++n) {
      float kr = g_kx[nk] * float(g_x[n]) + g_ky[nk] * float(g_y[n]) + g_kz[nk] * float(g_z[n]);
      const float charge = g_charge[n];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      S_real += charge * cos_kr;
      S_imag -= charge * sin_kr;
    }
    g_S_real[nk] = S_real;
    g_S_imag[nk] = S_imag;
  }
}

static __global__ void find_force_charge_reciprocal_space(
  const int N,
  const int N1,
  const int N2,
  const int num_kpoints,
  const float alpha_factor,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const float q = g_charge[n];
    float temp_energy_sum = 0.0f;
    float temp_virial_sum[6] = {0.0f};
    float temp_force_sum[3] = {0.0f};
    float temp_D_real_sum = 0.0f;
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const float kx = g_kx[nk];
      const float ky = g_ky[nk];
      const float kz = g_kz[nk];
      const float kr = kx * g_x[n] + ky * g_y[n] + kz * g_z[n];
      const float G = g_G[nk];
      const float S_real = g_S_real[nk];
      const float S_imag = g_S_imag[nk];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
      const float GSE = G * (S_real * cos_kr - S_imag * sin_kr);
      const float qGSE = q * GSE;
      temp_energy_sum += qGSE;
      const float alpha_k_factor = 2.0f * alpha_factor + 2.0f / (kx * kx + ky * ky + kz * kz);
      temp_virial_sum[0] += qGSE * (1.0f - alpha_k_factor * kx * kx); // xx
      temp_virial_sum[1] += qGSE * (1.0f - alpha_k_factor * ky * ky); // yy
      temp_virial_sum[2] += qGSE * (1.0f - alpha_k_factor * kz * kz); // zz
      temp_virial_sum[3] -= qGSE * (alpha_k_factor * kx * ky); // xy
      temp_virial_sum[4] -= qGSE * (alpha_k_factor * ky * kz); // yz
      temp_virial_sum[5] -= qGSE * (alpha_k_factor * kz * kx); // zx
      temp_D_real_sum += GSE;
      temp_force_sum[0] += kx * imag_term;
      temp_force_sum[1] += ky * imag_term;
      temp_force_sum[2] += kz * imag_term;
    }
    g_pe[n] += K_C_SP * temp_energy_sum;
    g_virial[n + 0 * N] += K_C_SP * temp_virial_sum[0];
    g_virial[n + 1 * N] += K_C_SP * temp_virial_sum[1];
    g_virial[n + 2 * N] += K_C_SP * temp_virial_sum[2];
    g_virial[n + 3 * N] += K_C_SP * temp_virial_sum[3];
    g_virial[n + 4 * N] += K_C_SP * temp_virial_sum[5];
    g_virial[n + 5 * N] += K_C_SP * temp_virial_sum[4];
    g_virial[n + 6 * N] += K_C_SP * temp_virial_sum[3];
    g_virial[n + 7 * N] += K_C_SP * temp_virial_sum[5];
    g_virial[n + 8 * N] += K_C_SP * temp_virial_sum[4];
    g_D_real[n] = 2.0f * K_C_SP * temp_D_real_sum;
    const float charge_factor = K_C_SP * 2.0f * q;
    g_fx[n] += charge_factor * temp_force_sum[0];
    g_fy[n] += charge_factor * temp_force_sum[1];
    g_fz[n] += charge_factor * temp_force_sum[2];
  }
}

void Ewald::find_force(
  const int N,
  const int N1,
  const int N2,
  const double* box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<float>& D_real,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& potential_per_atom)
{
  find_k_and_G(box);
  find_structure_factor<<<(num_kpoints_max - 1) / 64 + 1, 64>>>(
    num_kpoints_max,
    N1,
    N2,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    kx.data(),
    ky.data(),
    kz.data(),
    S_real.data(),
    S_imag.data());
  GPU_CHECK_KERNEL

  find_force_charge_reciprocal_space<<<(N - 1) / 64 + 1, 64>>>(
    N,
    N1,
    N2,
    num_kpoints_max,
    alpha_factor,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    S_real.data(),
    S_imag.data(),
    D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data(),
    potential_per_atom.data());
  GPU_CHECK_KERNEL
}
