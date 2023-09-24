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

#include "hamiltonian.cuh"
#include "model.cuh"
#include "vector.cuh"
#include <string.h>    // memcpy
#define BLOCK_SIZE 512 // optimized

#ifndef CPU_ONLY
void Hamiltonian::initialize_gpu(Model& model)
{
  n = model.number_of_atoms;
  max_neighbor = model.max_neighbor;
  energy_max = model.energy_max;
  grid_size = (model.number_of_atoms - 1) / BLOCK_SIZE + 1;

  CHECK(cudaMalloc((void**)&neighbor_number, sizeof(int) * n));
  CHECK(cudaMalloc((void**)&neighbor_list, sizeof(int) * model.number_of_pairs));
  CHECK(cudaMalloc((void**)&potential, sizeof(real) * n));
  CHECK(cudaMalloc((void**)&hopping_real, sizeof(real) * model.number_of_pairs));
  CHECK(cudaMalloc((void**)&hopping_imag, sizeof(real) * model.number_of_pairs));
  CHECK(cudaMalloc((void**)&xx, sizeof(real) * model.number_of_pairs));

  CHECK(
    cudaMemcpy(neighbor_number, model.neighbor_number, sizeof(int) * n, cudaMemcpyHostToDevice));
  delete[] model.neighbor_number;
  CHECK(cudaMemcpy(potential, model.potential, sizeof(real) * n, cudaMemcpyHostToDevice));
  delete[] model.potential;

  int* neighbor_list_new = new int[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      neighbor_list_new[m * n + i] = model.neighbor_list[i * max_neighbor + m];
    }
  }
  delete[] model.neighbor_list;
  CHECK(cudaMemcpy(
    neighbor_list, neighbor_list_new, sizeof(int) * model.number_of_pairs, cudaMemcpyHostToDevice));
  delete[] neighbor_list_new;

  real* hopping_real_new = new real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      hopping_real_new[m * n + i] = model.hopping_real[i * max_neighbor + m];
    }
  }
  delete[] model.hopping_real;
  CHECK(cudaMemcpy(
    hopping_real, hopping_real_new, sizeof(real) * model.number_of_pairs, cudaMemcpyHostToDevice));
  delete[] hopping_real_new;

  real* hopping_imag_new = new real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      hopping_imag_new[m * n + i] = model.hopping_imag[i * max_neighbor + m];
    }
  }
  delete[] model.hopping_imag;
  CHECK(cudaMemcpy(
    hopping_imag, hopping_imag_new, sizeof(real) * model.number_of_pairs, cudaMemcpyHostToDevice));
  delete[] hopping_imag_new;

  real* xx_new = new real[model.number_of_pairs];
  for (int m = 0; m < max_neighbor; ++m) {
    for (int i = 0; i < n; ++i) {
      xx_new[m * n + i] = model.xx[i * max_neighbor + m];
    }
  }
  delete[] model.xx;
  CHECK(cudaMemcpy(xx, xx_new, sizeof(real) * model.number_of_pairs, cudaMemcpyHostToDevice));
  delete[] xx_new;
}
#else
void Hamiltonian::initialize_cpu(Model& model)
{
  n = model.number_of_atoms;
  max_neighbor = model.max_neighbor;
  energy_max = model.energy_max;
  int number_of_pairs = model.number_of_pairs;

  neighbor_number = new int[n];
  memcpy(neighbor_number, model.neighbor_number, sizeof(int) * n);
  delete[] model.neighbor_number;

  neighbor_list = new int[number_of_pairs];
  memcpy(neighbor_list, model.neighbor_list, sizeof(int) * number_of_pairs);
  delete[] model.neighbor_list;

  potential = new real[n];
  memcpy(potential, model.potential, sizeof(real) * n);
  delete[] model.potential;

  hopping_real = new real[number_of_pairs];
  memcpy(hopping_real, model.hopping_real, sizeof(real) * number_of_pairs);
  delete[] model.hopping_real;

  hopping_imag = new real[number_of_pairs];
  memcpy(hopping_imag, model.hopping_imag, sizeof(real) * number_of_pairs);
  delete[] model.hopping_imag;

  xx = new real[number_of_pairs];
  memcpy(xx, model.xx, sizeof(real) * number_of_pairs);
  delete[] model.xx;
}
#endif

Hamiltonian::Hamiltonian(Model& model)
{
#ifndef CPU_ONLY
  initialize_gpu(model);
#else
  initialize_cpu(model);
#endif
}

Hamiltonian::~Hamiltonian()
{
#ifndef CPU_ONLY
  CHECK(cudaFree(neighbor_number));
  CHECK(cudaFree(neighbor_list));
  CHECK(cudaFree(potential));
  CHECK(cudaFree(hopping_real));
  CHECK(cudaFree(hopping_imag));
  CHECK(cudaFree(xx));
#else
  delete[] neighbor_number;
  delete[] neighbor_list;
  delete[] potential;
  delete[] hopping_real;
  delete[] hopping_imag;
  delete[] xx;
#endif
}

#ifndef CPU_ONLY
__global__ void gpu_apply_hamiltonian(
  int number_of_atoms,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_in_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_in_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    g_state_out_real[n] = temp_real;
    g_state_out_imag[n] = temp_imag;
  }
}
#else
void cpu_apply_hamiltonian(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_in_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_in_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    g_state_out_real[n] = temp_real;
    g_state_out_imag[n] = temp_imag;
  }
}
#endif

// |output> = H |input>
void Hamiltonian::apply(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
  gpu_apply_hamiltonian<<<grid_size, BLOCK_SIZE>>>(
    n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
    input.real_part, input.imag_part, output.real_part, output.imag_part);
  CHECK(cudaGetLastError());
#else
  cpu_apply_hamiltonian(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
__global__ void gpu_apply_commutator(
  int number_of_atoms,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      real xx = g_xx[index_1];
      temp_real -= (a * c - b * d) * xx;
      temp_imag -= (a * d + b * c) * xx;
    }
    g_state_out_real[n] = temp_real / energy_max; // scale
    g_state_out_imag[n] = temp_imag / energy_max; // scale
  }
}
#else
void cpu_apply_commutator(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      real xx = g_xx[index_1];
      temp_real -= (a * c - b * d) * xx;
      temp_imag -= (a * d + b * c) * xx;
    }
    g_state_out_real[n] = temp_real / energy_max; // scale
    g_state_out_imag[n] = temp_imag / energy_max; // scale
  }
}
#endif

// |output> = [X, H] |input>
void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
  gpu_apply_commutator<<<grid_size, BLOCK_SIZE>>>(
    n, energy_max, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx, input.real_part,
    input.imag_part, output.real_part, output.imag_part);
  CHECK(cudaGetLastError());
#else
  cpu_apply_commutator(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx,
    input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

#ifndef CPU_ONLY
__global__ void gpu_apply_current(
  int number_of_atoms,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += (a * c - b * d) * g_xx[index_1];
      temp_imag += (a * d + b * c) * g_xx[index_1];
    }
    g_state_out_real[n] = +temp_imag;
    g_state_out_imag[n] = -temp_real;
  }
}
#else
void cpu_apply_current(
  int number_of_atoms,
  int max_neighbor,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_in_real,
  real* g_state_in_imag,
  real* g_state_out_real,
  real* g_state_out_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = 0.0;
    real temp_imag = 0.0;
    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_in_real[index_2];
      real d = g_state_in_imag[index_2];
      temp_real += (a * c - b * d) * g_xx[index_1];
      temp_imag += (a * d + b * c) * g_xx[index_1];
    }
    g_state_out_real[n] = +temp_imag;
    g_state_out_imag[n] = -temp_real;
  }
}
#endif

// |output> = V |input>
void Hamiltonian::apply_current(Vector& input, Vector& output)
{
#ifndef CPU_ONLY
  gpu_apply_current<<<grid_size, BLOCK_SIZE>>>(
    n, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx, input.real_part,
    input.imag_part, output.real_part, output.imag_part);
  CHECK(cudaGetLastError());
#else
  cpu_apply_current(
    n, max_neighbor, neighbor_number, neighbor_list, hopping_real, hopping_imag, xx,
    input.real_part, input.imag_part, output.real_part, output.imag_part);
#endif
}

// Kernel which calculates the two first terms of time evolution as described by
// Eq. (36) in [Comput. Phys. Commun.185, 28 (2014)].
#ifndef CPU_ONLY
__global__ void gpu_chebyshev_01(
  int number_of_atoms,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_real,
  real* g_state_imag,
  real b0,
  real b1,
  int direction)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real bessel_0 = b0;
    real bessel_1 = b1 * direction;
    g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
    g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
  }
}
#else
void cpu_chebyshev_01(
  int number_of_atoms,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_real,
  real* g_state_imag,
  real b0,
  real b1,
  int direction)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real bessel_0 = b0;
    real bessel_1 = b1 * direction;
    g_state_real[n] = bessel_0 * g_state_0_real[n] + bessel_1 * g_state_1_imag[n];
    g_state_imag[n] = bessel_0 * g_state_0_imag[n] - bessel_1 * g_state_1_real[n];
  }
}
#endif

// Wrapper for the kernel above
void Hamiltonian::chebyshev_01(
  Vector& state_0, Vector& state_1, Vector& state, real bessel_0, real bessel_1, int direction)
{
#ifndef CPU_ONLY
  gpu_chebyshev_01<<<grid_size, BLOCK_SIZE>>>(
    n, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state.real_part,
    state.imag_part, bessel_0, bessel_1, direction);
  CHECK(cudaGetLastError());
#else
  cpu_chebyshev_01(
    n, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state.real_part,
    state.imag_part, bessel_0, bessel_1, direction);
#endif
}

// Kernel for calculating further terms of Eq. (36)
// in [Comput. Phys. Commun.185, 28 (2014)].
#ifndef CPU_ONLY
__global__ void gpu_chebyshev_2(
  int number_of_atoms,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_2_real,
  real* g_state_2_imag,
  real* g_state_real,
  real* g_state_imag,
  real bessel_m,
  int label)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    switch (label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_real;
        g_state_imag[n] += bessel_m * temp_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_real;
        g_state_imag[n] -= bessel_m * temp_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_imag;
        g_state_imag[n] -= bessel_m * temp_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_imag;
        g_state_imag[n] += bessel_m * temp_real;
        break;
      }
    }
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#else
void cpu_chebyshev_2(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_2_real,
  real* g_state_2_imag,
  real* g_state_real,
  real* g_state_imag,
  real bessel_m,
  int label)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }
    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    switch (label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_real;
        g_state_imag[n] += bessel_m * temp_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_real;
        g_state_imag[n] -= bessel_m * temp_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_imag;
        g_state_imag[n] -= bessel_m * temp_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_imag;
        g_state_imag[n] += bessel_m * temp_real;
        break;
      }
    }
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#endif

// Wrapper for the kernel above
void Hamiltonian::chebyshev_2(
  Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, real bessel_m, int label)
{
#ifndef CPU_ONLY
  gpu_chebyshev_2<<<grid_size, BLOCK_SIZE>>>(
    n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
    state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state_2.real_part,
    state_2.imag_part, state.real_part, state.imag_part, bessel_m, label);
  CHECK(cudaGetLastError());
#else
  cpu_chebyshev_2(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part,
    state_2.real_part, state_2.imag_part, state.real_part, state.imag_part, bessel_m, label);
#endif
}

// Kernel which calculates the two first terms of commutator [X, U(dt)]
// Corresponds to Eq. (37) in [Comput. Phys. Commun.185, 28 (2014)].
#ifndef CPU_ONLY
__global__ void gpu_chebyshev_1x(
  int number_of_atoms,
  real* g_state_1x_real,
  real* g_state_1x_imag,
  real* g_state_real,
  real* g_state_imag,
  real g_bessel_1)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real b1 = g_bessel_1;
    g_state_real[n] = +b1 * g_state_1x_imag[n];
    g_state_imag[n] = -b1 * g_state_1x_real[n];
  }
}
#else
void cpu_chebyshev_1x(
  int number_of_atoms,
  real* g_state_1x_real,
  real* g_state_1x_imag,
  real* g_state_real,
  real* g_state_imag,
  real g_bessel_1)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real b1 = g_bessel_1;
    g_state_real[n] = +b1 * g_state_1x_imag[n];
    g_state_imag[n] = -b1 * g_state_1x_real[n];
  }
}
#endif

// Wrapper for kernel above
void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, real bessel_1)
{
#ifndef CPU_ONLY
  gpu_chebyshev_1x<<<grid_size, BLOCK_SIZE>>>(
    n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
  CHECK(cudaGetLastError());
#else
  cpu_chebyshev_1x(
    n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
#endif
}

// Kernel which calculates the further terms of [X, U(dt)]
#ifndef CPU_ONLY
__global__ void gpu_chebyshev_2x(
  int number_of_atoms,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_0x_real,
  real* g_state_0x_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_1x_real,
  real* g_state_1x_imag,
  real* g_state_2_real,
  real* g_state_2_imag,
  real* g_state_2x_real,
  real* g_state_2x_imag,
  real* g_state_real,
  real* g_state_imag,
  real g_bessel_m,
  int g_label)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_1_real[n];    // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n];    // on-site
    real temp_x_real = g_potential[n] * g_state_1x_real[n]; // on-site
    real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];

      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping

      real cx = g_state_1x_real[index_2];
      real dx = g_state_1x_imag[index_2];
      temp_x_real += a * cx - b * dx; // hopping
      temp_x_imag += a * dx + b * cx; // hopping

      real xx = g_xx[index_1];
      temp_x_real -= (a * c - b * d) * xx; // hopping
      temp_x_imag -= (a * d + b * c) * xx; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;

    temp_x_real /= energy_max; // scale
    temp_x_imag /= energy_max; // scale
    temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
    temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
    g_state_2x_real[n] = temp_x_real;
    g_state_2x_imag[n] = temp_x_imag;

    real bessel_m = g_bessel_m;
    switch (g_label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_x_real;
        g_state_imag[n] += bessel_m * temp_x_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_x_real;
        g_state_imag[n] -= bessel_m * temp_x_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_x_imag;
        g_state_imag[n] -= bessel_m * temp_x_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_x_imag;
        g_state_imag[n] += bessel_m * temp_x_real;
        break;
      }
    }
  }
}
#else
void cpu_chebyshev_2x(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_0x_real,
  real* g_state_0x_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_1x_real,
  real* g_state_1x_imag,
  real* g_state_2_real,
  real* g_state_2_imag,
  real* g_state_2x_real,
  real* g_state_2x_imag,
  real* g_state_real,
  real* g_state_imag,
  real g_bessel_m,
  int g_label)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_1_real[n];    // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n];    // on-site
    real temp_x_real = g_potential[n] * g_state_1x_real[n]; // on-site
    real temp_x_imag = g_potential[n] * g_state_1x_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];

      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping

      real cx = g_state_1x_real[index_2];
      real dx = g_state_1x_imag[index_2];
      temp_x_real += a * cx - b * dx; // hopping
      temp_x_imag += a * dx + b * cx; // hopping

      real xx = g_xx[index_1];
      temp_x_real -= (a * c - b * d) * xx; // hopping
      temp_x_imag -= (a * d + b * c) * xx; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale
    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;

    temp_x_real /= energy_max; // scale
    temp_x_imag /= energy_max; // scale
    temp_x_real = 2.0 * temp_x_real - g_state_0x_real[n];
    temp_x_imag = 2.0 * temp_x_imag - g_state_0x_imag[n];
    g_state_2x_real[n] = temp_x_real;
    g_state_2x_imag[n] = temp_x_imag;

    real bessel_m = g_bessel_m;
    switch (g_label) {
      case 1: {
        g_state_real[n] += bessel_m * temp_x_real;
        g_state_imag[n] += bessel_m * temp_x_imag;
        break;
      }
      case 2: {
        g_state_real[n] -= bessel_m * temp_x_real;
        g_state_imag[n] -= bessel_m * temp_x_imag;
        break;
      }
      case 3: {
        g_state_real[n] += bessel_m * temp_x_imag;
        g_state_imag[n] -= bessel_m * temp_x_real;
        break;
      }
      case 4: {
        g_state_real[n] -= bessel_m * temp_x_imag;
        g_state_imag[n] += bessel_m * temp_x_real;
        break;
      }
    }
  }
}
#endif

// Wrapper for the kernel above
void Hamiltonian::chebyshev_2x(
  Vector& state_0,
  Vector& state_0x,
  Vector& state_1,
  Vector& state_1x,
  Vector& state_2,
  Vector& state_2x,
  Vector& state,
  real bessel_m,
  int label)
{
#ifndef CPU_ONLY
  gpu_chebyshev_2x<<<grid_size, BLOCK_SIZE>>>(
    n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag, xx,
    state_0.real_part, state_0.imag_part, state_0x.real_part, state_0x.imag_part, state_1.real_part,
    state_1.imag_part, state_1x.real_part, state_1x.imag_part, state_2.real_part, state_2.imag_part,
    state_2x.real_part, state_2x.imag_part, state.real_part, state.imag_part, bessel_m, label);
  CHECK(cudaGetLastError());
#else
  cpu_chebyshev_2x(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, xx, state_0.real_part, state_0.imag_part, state_0x.real_part, state_0x.imag_part,
    state_1.real_part, state_1.imag_part, state_1x.real_part, state_1x.imag_part, state_2.real_part,
    state_2.imag_part, state_2x.real_part, state_2x.imag_part, state.real_part, state.imag_part,
    bessel_m, label);
#endif
}

// Kernel for doing the Chebyshev iteration phi_2 = 2 * H * phi_1 - phi_0.
#ifndef CPU_ONLY
__global__ void gpu_kernel_polynomial(
  int number_of_atoms,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_2_real,
  real* g_state_2_imag)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = m * number_of_atoms + n;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#else
void cpu_kernel_polynomial(
  int number_of_atoms,
  int max_neighbor,
  real energy_max,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_state_0_real,
  real* g_state_0_imag,
  real* g_state_1_real,
  real* g_state_1_imag,
  real* g_state_2_real,
  real* g_state_2_imag)
{
  for (int n = 0; n < number_of_atoms; ++n) {
    real temp_real = g_potential[n] * g_state_1_real[n]; // on-site
    real temp_imag = g_potential[n] * g_state_1_imag[n]; // on-site

    for (int m = 0; m < g_neighbor_number[n]; ++m) {
      int index_1 = n * max_neighbor + m;
      int index_2 = g_neighbor_list[index_1];
      real a = g_hopping_real[index_1];
      real b = g_hopping_imag[index_1];
      real c = g_state_1_real[index_2];
      real d = g_state_1_imag[index_2];
      temp_real += a * c - b * d; // hopping
      temp_imag += a * d + b * c; // hopping
    }

    temp_real /= energy_max; // scale
    temp_imag /= energy_max; // scale

    temp_real = 2.0 * temp_real - g_state_0_real[n];
    temp_imag = 2.0 * temp_imag - g_state_0_imag[n];
    g_state_2_real[n] = temp_real;
    g_state_2_imag[n] = temp_imag;
  }
}
#endif

// Wrapper for the Chebyshev iteration
void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
#ifndef CPU_ONLY
  gpu_kernel_polynomial<<<grid_size, BLOCK_SIZE>>>(
    n, energy_max, neighbor_number, neighbor_list, potential, hopping_real, hopping_imag,
    state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part, state_2.real_part,
    state_2.imag_part);
  CHECK(cudaGetLastError());
#else
  cpu_kernel_polynomial(
    n, max_neighbor, energy_max, neighbor_number, neighbor_list, potential, hopping_real,
    hopping_imag, state_0.real_part, state_0.imag_part, state_1.real_part, state_1.imag_part,
    state_2.real_part, state_2.imag_part);
#endif
}
