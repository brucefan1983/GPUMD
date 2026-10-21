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

#include "nep_long_range.cuh"
#include "main_nep/dataset.cuh"
#include "nep_charge_dataset.cuh"
#include "nep_vdw_parameters.cuh"
#include "utilities/error.cuh"
#include <climits>
#include <stdexcept>
#include <vector>

static __global__ void find_structure_factor(
  const int* Na,
  const int* Na_sum,
  const float* g_weight,
  const float* g_x,
  const float* g_y,
  const float* g_z,
  const int* g_kpoint_offset,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  float* g_S_real,
  float* g_S_imag)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  const int kpoint_begin = g_kpoint_offset[blockIdx.x];
  const int num_kpoints = g_kpoint_offset[blockIdx.x + 1] - kpoint_begin;

  for (int nk = threadIdx.x; nk < num_kpoints; nk += blockDim.x) {
    const int nc_nk = kpoint_begin + nk;
    float S_real = 0.0f;
    float S_imag = 0.0f;
    for (int n = N1; n < N2; ++n) {
      float kr = g_kx[nc_nk] * g_x[n] + g_ky[nc_nk] * g_y[n] + g_kz[nc_nk] * g_z[n];
      const float weight = g_weight[n];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      S_real += weight * cos_kr;
      S_imag -= weight * sin_kr;
    }
    g_S_real[nc_nk] = S_real;
    g_S_imag[nc_nk] = S_imag;
  }
}

static __global__ void zero_total_charge(
  const int* Na,
  const int* Na_sum,
  const float* g_charge_ref,
  const float* g_charge,
  float* g_charge_shifted)
{
  int tid = threadIdx.x;
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      charge += g_charge[n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      g_charge_shifted[n] = g_charge[n] + (g_charge_ref[blockIdx.x] - s_charge[0]) / (N2 - N1);
    }
  }
}

// Per-structure mean subtraction of D_real for training.
// Chain rule correction for zero_total_charge (see MD-side comment for derivation).
// Launch: <<<Nc, 1024>>>
static __global__ void zero_mean_D_real_train(
  const int* Na,
  const int* Na_sum,
  float* g_D_real)
{
  int tid = threadIdx.x;
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      sum += (double)g_D_real[n];
    }
  }
  s_sum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  float mean_D = (float)(s_sum[0] / (N2 - N1));
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024 + N1;
    if (n < N2) {
      g_D_real[n] -= mean_D;
    }
  }
}

static __global__ void find_bec_diagonal(const int N, const float* g_q, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_bec[n1 + N * 0] = g_q[n1];
    g_bec[n1 + N * 1] = 0.0f;
    g_bec[n1 + N * 2] = 0.0f;
    g_bec[n1 + N * 3] = 0.0f;
    g_bec[n1 + N * 4] = g_q[n1];
    g_bec[n1 + N * 5] = 0.0f;
    g_bec[n1 + N * 6] = 0.0f;
    g_bec[n1 + N * 7] = 0.0f;
    g_bec[n1 + N * 8] = g_q[n1];
  }
}

static __global__ void scale_bec(const int N, const float* sqrt_epsilon_inf, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

static __global__ void find_force_vdw_long_range(
  const int N,
  const float alpha,
  const int* Na,
  const int* Na_sum,
  const int* g_pbc,
  const float* g_C6,
  const float* g_x,
  const float* g_y,
  const float* g_z,
  const int* g_kpoint_offset,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G_vdw,
  const float* g_G_vdw_virial,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_C6,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  double* g_pe)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int number_of_batches = (N2 - N1 - 1) / 1024 + 1;
  const int is_periodic = g_pbc[blockIdx.x];
  const int kpoint_begin = g_kpoint_offset[blockIdx.x];
  const int num_kpoints = g_kpoint_offset[blockIdx.x + 1] - kpoint_begin;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = threadIdx.x + batch * 1024 + N1;
    if (n < N2) {
      const float C6_1 = g_C6[n];
      const float alpha_squared = alpha * alpha;
      const float alpha_sixth =
        alpha_squared * alpha_squared * alpha_squared;
      if (!is_periodic) {
        double temp_energy_sum = 0.0;
        float temp_D_C6_sum = 0.0f;
        float temp_virial_sum[6] = {0.0f};
        float temp_force_sum[3] = {0.0f};
        for (int n2 = N1; n2 < N2; ++n2) {
          if (n2 == n) {
            continue;
          }
          const float x12 = g_x[n2] - g_x[n];
          const float y12 = g_y[n2] - g_y[n];
          const float z12 = g_z[n2] - g_z[n];
          const float distance_square =
            x12 * x12 + y12 * y12 + z12 * z12;
          float potential;
          float force_factor;
          find_vdw_long_range(
            alpha_squared,
            alpha_sixth,
            distance_square,
            potential,
            force_factor);
          const float C6_2 = g_C6[n2];
          const float C6_product = C6_1 * C6_2;
          const float f12[3] = {
            x12 * C6_product * force_factor,
            y12 * C6_product * force_factor,
            z12 * C6_product * force_factor};
          temp_energy_sum += 0.5 * C6_product * potential;
          temp_D_C6_sum += C6_2 * potential;
          temp_force_sum[0] += f12[0];
          temp_force_sum[1] += f12[1];
          temp_force_sum[2] += f12[2];
          temp_virial_sum[0] -= 0.5f * x12 * f12[0];
          temp_virial_sum[1] -= 0.5f * y12 * f12[1];
          temp_virial_sum[2] -= 0.5f * z12 * f12[2];
          temp_virial_sum[3] -= 0.5f * x12 * f12[1];
          temp_virial_sum[4] -= 0.5f * y12 * f12[2];
          temp_virial_sum[5] -= 0.5f * z12 * f12[0];
        }
        g_pe[n] += temp_energy_sum;
        for (int d = 0; d < 6; ++d) {
          g_virial[n + N * d] += temp_virial_sum[d];
        }
        g_D_C6[n] = temp_D_C6_sum;
        g_fx[n] += temp_force_sum[0];
        g_fy[n] += temp_force_sum[1];
        g_fz[n] += temp_force_sum[2];
        continue;
      }

      double temp_energy_sum = 0.0;
      float temp_virial_sum[6] = {0.0f};
      float temp_force_sum[3] = {0.0f};
      float temp_D_C6_sum = 0.0f;
      for (int nk = 0; nk < num_kpoints; ++nk) {
        const int nc_nk = kpoint_begin + nk;
        const float kx = g_kx[nc_nk];
        const float ky = g_ky[nc_nk];
        const float kz = g_kz[nc_nk];
        const float kr = kx * g_x[n] + ky * g_y[n] + kz * g_z[n];
        const float G = g_G_vdw[nc_nk];
        const float H = g_G_vdw_virial[nc_nk];
        const float S_real = g_S_real[nc_nk];
        const float S_imag = g_S_imag[nc_nk];
        float sin_kr = sin(kr);
        float cos_kr = cos(kr);
        const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
        const float SS = S_real * S_real + S_imag * S_imag;
        const float GSS = G * SS;
        const float HSS = H * SS;
        temp_energy_sum += GSS;
        temp_virial_sum[0] += GSS - HSS * kx * kx; // xx
        temp_virial_sum[1] += GSS - HSS * ky * ky; // yy
        temp_virial_sum[2] += GSS - HSS * kz * kz; // zz
        temp_virial_sum[3] -= HSS * kx * ky; // xy
        temp_virial_sum[4] -= HSS * ky * kz; // yz
        temp_virial_sum[5] -= HSS * kz * kx; // zx
        temp_D_C6_sum += G * (S_real * cos_kr - S_imag * sin_kr);
        temp_force_sum[0] += kx * imag_term;
        temp_force_sum[1] += ky * imag_term;
        temp_force_sum[2] += kz * imag_term;
      }
      // Remove the r=0 self interaction contained in the structure factor.
      g_pe[n] += temp_energy_sum / (N2 - N1) +
        double(alpha_sixth) * C6_1 * C6_1 / 12.0;
      for (int d = 0; d < 6; ++d) {
        g_virial[n + N * d] += temp_virial_sum[d] / (N2 - N1);
      }
      g_D_C6[n] = 2.0f * temp_D_C6_sum +
        alpha_sixth * C6_1 / 6.0f;
      const float C6_factor = 2.0f * C6_1;
      g_fx[n] += C6_factor * temp_force_sum[0];
      g_fy[n] += C6_factor * temp_force_sum[1];
      g_fz[n] += C6_factor * temp_force_sum[2];
    }
  }
}

void enforce_nep_total_charge(Dataset& dataset, NEP_Charge_Dataset& charge_dataset)
{
  zero_total_charge<<<dataset.Nc, 1024>>>(
    dataset.Na.data(), dataset.Na_sum.data(), charge_dataset.charge_ref_gpu.data(),
    charge_dataset.charge.data(), charge_dataset.charge_shifted.data());
  GPU_CHECK_KERNEL
}

void subtract_nep_charge_derivative_mean(const Dataset& dataset, float* D_real)
{
  zero_mean_D_real_train<<<dataset.Nc, 1024>>>(
    dataset.Na.data(), dataset.Na_sum.data(), D_real);
  GPU_CHECK_KERNEL
}

void initialize_nep_bec(Dataset& dataset, NEP_Charge_Dataset& charge_dataset)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;
  find_bec_diagonal<<<grid_size, block_size>>>(
    dataset.N, charge_dataset.charge_shifted.data(), charge_dataset.bec.data());
  GPU_CHECK_KERNEL
}

void scale_nep_bec(Dataset& dataset, NEP_Charge_Dataset& charge_dataset, const float* sqrt_epsilon_inf)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;
  scale_bec<<<grid_size, block_size>>>(dataset.N, sqrt_epsilon_inf, charge_dataset.bec.data());
  GPU_CHECK_KERNEL
}

void compute_nep_structure_factor(
  const Dataset& dataset,
  const float* weights,
  const int* kpoint_offset,
  const float* kx, const float* ky, const float* kz,
  float* S_real, float* S_imag)
{
  find_structure_factor<<<dataset.Nc, 1024>>>(
    dataset.Na.data(), dataset.Na_sum.data(), weights,
    dataset.r.data(), dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2,
    kpoint_offset, kx, ky, kz, S_real, S_imag);
  GPU_CHECK_KERNEL
}

void add_nep_vdw_long_range(
  Dataset& dataset,
  float alpha,
  const float* C6,
  const int* kpoint_offset,
  const float* kx, const float* ky, const float* kz,
  const float* G_vdw, const float* G_vdw_virial,
  const float* S_real, const float* S_imag,
  float* D_C6)
{
  find_force_vdw_long_range<<<dataset.Nc, 1024>>>(
    dataset.N, alpha,
    dataset.Na.data(), dataset.Na_sum.data(), dataset.pbc.data(), C6,
    dataset.r.data(), dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2,
    kpoint_offset, kx, ky, kz, G_vdw, G_vdw_virial, S_real, S_imag, D_C6,
    dataset.force.data(), dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2,
    dataset.virial.data(), dataset.energy.data());
  GPU_CHECK_KERNEL
}

int update_nep_kpoint_offsets(int Nc, GPU_Vector<int>& offsets)
{
  std::vector<int> kpoint_offset(Nc + 1);
  offsets.copy_to_host(kpoint_offset.data(), Nc);
  long long total_num_kpoints = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    const int num_kpoints = kpoint_offset[nc];
    kpoint_offset[nc] = static_cast<int>(total_num_kpoints);
    total_num_kpoints += num_kpoints;
    if (total_num_kpoints > INT_MAX) {
      throw std::runtime_error("The total number of Ewald K points is too large.");
    }
  }
  kpoint_offset[Nc] = static_cast<int>(total_num_kpoints);
  offsets.copy_from_host(kpoint_offset.data(), Nc + 1);
  return static_cast<int>(total_num_kpoints);
}
