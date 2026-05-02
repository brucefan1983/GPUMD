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
The FIRE (fast inertial relaxation engine) minimizer
Reference: PhysRevLett 97, 170201 (2006)
           Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire_box_change.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace
{

// C = A * B, 3x3 matrix
void matrix_multiply(const double* A, const double* B, double* C)
{
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      C[i * 3 + j] = 0.0;
      for (int k = 0; k < 3; ++k) {
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
      }
    }
  }
}

__global__ void
update_pos_with_strain_kernel(int N, double* pos, const double* v, const double* dEps, double dt)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x = pos[i];
    double y = pos[N + i];
    double z = pos[2 * N + i];

    // dr = dEps * r
    double dx_strain = dEps[0] * x + dEps[1] * y + dEps[2] * z;
    double dy_strain = dEps[3] * x + dEps[4] * y + dEps[5] * z;
    double dz_strain = dEps[6] * x + dEps[7] * y + dEps[8] * z;

    pos[i] = x + v[i] * dt + dx_strain;
    pos[N + i] = y + v[N + i] * dt + dy_strain;
    pos[2 * N + i] = z + v[2 * N + i] * dt + dz_strain;
  }
}

static __global__ void gpu_sum_virial(
  int N,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_s)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_s[1024];
  double s = 0.0;

  switch (bid) {
    case 0:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_sxx[n];
      }
      break;
    case 1:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_sxy[n];
      }
      break;
    case 2:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_sxz[n];
      }
      break;
    case 3:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_syx[n];
      }
      break;
    case 4:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_syy[n];
      }
      break;
    case 5:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_syz[n];
      }
      break;
    case 6:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_szx[n];
      }
      break;
    case 7:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_szy[n];
      }
      break;
    case 8:
      for (int p = 0; p < number_of_patches; ++p) {
        int n = tid + p * 1024;
        if (n < N)
          s += g_szz[n];
      }
      break;
  }
  s_s[tid] = s;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_s[tid] += s_s[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    g_s[bid] = s_s[0];
}

__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

__global__ void gpu_vector_sum(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] + b[n];
}

__global__ void gpu_pairwise_product(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] * b[n];
}

void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

__global__ void gpu_sum(const int size, double* a, double* result)
{
  int number_of_patches = (size - 1) / 1024 + 1;
  int tid = threadIdx.x;
  int n, patch;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * 1024;
    if (n < size)
      data[tid] += a[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      data[tid] += data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    *result = data[0];
}

double sum(GPU_Vector<double>& a)
{
  double ret;
  GPU_Vector<double> result(1);
  gpu_sum<<<1, 1024>>>(a.size(), a.data(), result.data());
  result.copy_to_host(&ret);
  return ret;
}

double dot(GPU_Vector<double>& a, GPU_Vector<double>& b)
{
  GPU_Vector<double> temp(a.size());
  pairwise_product(a, b, temp);
  return sum(temp);
}

void scalar_multiply(const double& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = b.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(size, a, b.data(), c.data());
}

void vector_sum(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_vector_sum<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}
} // namespace

void Minimizer_FIRE_Box_Change::compute(
  Force& force,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& position_per_atom,
  std::vector<Group>& group)
{
  double next_dt;
  const int size = number_of_atoms_ * 3;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;

  GPU_Vector<double> v(size + 9, 0.0);
  GPU_Vector<double> temp1(size + 9);
  GPU_Vector<double> temp2(size + 9);
  GPU_Vector<double> force_temp(size + 9, 0.0);
  GPU_Vector<double> virialtot(9);

  double L_scale = std::cbrt(box.get_volume());
  const double stress_tolerance_GPa = 1e-4;

  printf("\nEnergy minimization with changed box started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {

    force.compute(
      box, position_per_atom, atom.type, group, atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);

    gpu_sum_virial<<<9, 1024>>>(
      number_of_atoms_,
      atom.virial_per_atom.data() + 0 * number_of_atoms_, // xx
      atom.virial_per_atom.data() + 3 * number_of_atoms_, // xy
      atom.virial_per_atom.data() + 4 * number_of_atoms_, // xz
      atom.virial_per_atom.data() + 6 * number_of_atoms_, // yx
      atom.virial_per_atom.data() + 1 * number_of_atoms_, // yy
      atom.virial_per_atom.data() + 5 * number_of_atoms_, // yz
      atom.virial_per_atom.data() + 7 * number_of_atoms_, // zx
      atom.virial_per_atom.data() + 8 * number_of_atoms_, // zy
      atom.virial_per_atom.data() + 2 * number_of_atoms_, // zz
      virialtot.data());
    GPU_CHECK_KERNEL

    double virial_cpu[9];
    virialtot.copy_to_host(virial_cpu);

    double current_volume = box.get_volume();
    double stress_GPa[9];
    double max_stress_component = 0.0;

    for (int i = 0; i < 9; i++) {
      stress_GPa[i] = virial_cpu[i] / current_volume * 160.2176621;
    }

    double pressure = (stress_GPa[0] + stress_GPa[4] + stress_GPa[8]) / 3.0;

    if (hydrostatic_strain == 1) {
      max_stress_component = std::abs(pressure);
      double trace_virial = virial_cpu[0] + virial_cpu[4] + virial_cpu[8];
      for (int i = 0; i < 9; i++)
        virial_cpu[i] = 0.0;
      virial_cpu[0] = trace_virial / 3.0;
      virial_cpu[4] = trace_virial / 3.0;
      virial_cpu[8] = trace_virial / 3.0;
    } else {
      for (int i = 0; i < 9; i++) {
        max_stress_component = std::max(max_stress_component, std::abs(stress_GPa[i]));
      }
    }

    for (int i = 0; i < 9; i++) {
      virial_cpu[i] = virial_cpu[i] / L_scale;
    }

    gpuMemcpy(
      force_temp.data(), atom.force_per_atom.data(), size * sizeof(double), gpuMemcpyDeviceToDevice);
    gpuMemcpy(force_temp.data() + size, virial_cpu, 9 * sizeof(double), gpuMemcpyHostToDevice);

    calculate_force_square_max(atom.force_per_atom);
    const double force_max_atom = sqrt(cpu_force_square_max_[0]);
    calculate_total_potential(atom.potential_per_atom);

    if (
      step == 0 || (step + 1) % base == 0 ||
      (force_max_atom < force_tolerance_ && max_stress_component < stress_tolerance_GPa)) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max_atom = %.10f eV/A, pressure = %.6f GPa.\n",
        step == 0 ? 0 : (step + 1),
        cpu_total_potential_[0],
        force_max_atom,
        pressure);

      if (force_max_atom < force_tolerance_ && max_stress_component < stress_tolerance_GPa) {
        printf(
          "  Converged! f_max_atom = %.2e < %.2e AND max_stress = %.2e GPa < %.2e GPa\n",
          force_max_atom,
          force_tolerance_,
          max_stress_component,
          stress_tolerance_GPa);
        break;
      }
    }

    P = dot(v, force_temp);

    if (P > 0) {
      if (N_pos > N_min) {
        next_dt = dt * f_inc;
        if (next_dt > dt_max)
          next_dt = dt_max;
        dt = next_dt;
        alpha *= f_alpha;
      }
      N_pos++;
    } else {
      next_dt = dt * f_dec;
      if (next_dt < dt_min)
        next_dt = dt_min;
      dt = next_dt;
      alpha = alpha_start;
      v.fill(0);
      N_pos = 0;
    }

    // MD Step - Implicit Euler Integration for Generalized Vector
    double F_modulus = sqrt(dot(force_temp, force_temp));
    double v_modulus = sqrt(dot(v, v));

    // dv = F/m*dt
    scalar_multiply(dt / m, force_temp, temp2);
    vector_sum(v, temp2, v);
    scalar_multiply(1 - alpha, v, temp1);

    if (F_modulus > 1e-12) {
      scalar_multiply(alpha * v_modulus / F_modulus, force_temp, temp2);
    } else {
      temp2.fill(0.0);
    }
    vector_sum(temp1, temp2, v);

    double v_box_cpu[9];
    gpuMemcpy(v_box_cpu, v.data() + size, 9 * sizeof(double), gpuMemcpyDeviceToHost);

    // dEps = v_box * dt / L_scale
    double dEps[9];
    for (int i = 0; i < 9; i++)
      dEps[i] = v_box_cpu[i] * dt / L_scale;

    // 1. H_new = H + dEps * H
    double dH[9];
    matrix_multiply(dEps, box.cpu_h, dH);
    for (int i = 0; i < 9; i++)
      box.cpu_h[i] += dH[i];
    box.get_inverse();

    // 2. dr = v*dt + dEps*r
    double* d_dEps;
    gpuMalloc(&d_dEps, 9 * sizeof(double));
    gpuMemcpy(d_dEps, dEps, 9 * sizeof(double), gpuMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (number_of_atoms_ - 1) / threadsPerBlock + 1;
    update_pos_with_strain_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      number_of_atoms_, position_per_atom.data(), v.data(), d_dEps, dt);

    gpuFree(d_dEps);
  }

  printf("Energy minimization finished.\n\n");
}