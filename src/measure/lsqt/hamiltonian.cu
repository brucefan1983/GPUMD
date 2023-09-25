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
#include "utilities/error.cuh"
#include "vector.cuh"
#include <string.h>    // memcpy
#define BLOCK_SIZE 512 // optimized

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
}

Hamiltonian::Hamiltonian(Model& model) { initialize_gpu(model); }

Hamiltonian::~Hamiltonian()
{
  CHECK(cudaFree(neighbor_number));
  CHECK(cudaFree(neighbor_list));
  CHECK(cudaFree(potential));
  CHECK(cudaFree(hopping_real));
  CHECK(cudaFree(hopping_imag));
  CHECK(cudaFree(xx));
}

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

// |output> = H |input>
void Hamiltonian::apply(Vector& input, Vector& output)
{
  gpu_apply_hamiltonian<<<grid_size, BLOCK_SIZE>>>(
    n,
    energy_max,
    neighbor_number,
    neighbor_list,
    potential,
    hopping_real,
    hopping_imag,
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part);
  CHECK(cudaGetLastError());
}

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

// |output> = [X, H] |input>
void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
  gpu_apply_commutator<<<grid_size, BLOCK_SIZE>>>(
    n,
    energy_max,
    neighbor_number,
    neighbor_list,
    hopping_real,
    hopping_imag,
    xx,
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part);
  CHECK(cudaGetLastError());
}

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

// |output> = V |input>
void Hamiltonian::apply_current(Vector& input, Vector& output)
{
  gpu_apply_current<<<grid_size, BLOCK_SIZE>>>(
    n,
    neighbor_number,
    neighbor_list,
    hopping_real,
    hopping_imag,
    xx,
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part);
  CHECK(cudaGetLastError());
}

// Kernel which calculates the two first terms of time evolution as described by
// Eq. (36) in [Comput. Phys. Commun.185, 28 (2014)].
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

// Wrapper for the kernel above
void Hamiltonian::chebyshev_01(
  Vector& state_0, Vector& state_1, Vector& state, real bessel_0, real bessel_1, int direction)
{
  gpu_chebyshev_01<<<grid_size, BLOCK_SIZE>>>(
    n,
    state_0.real_part,
    state_0.imag_part,
    state_1.real_part,
    state_1.imag_part,
    state.real_part,
    state.imag_part,
    bessel_0,
    bessel_1,
    direction);
  CHECK(cudaGetLastError());
}

// Kernel for calculating further terms of Eq. (36)
// in [Comput. Phys. Commun.185, 28 (2014)].
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

// Wrapper for the kernel above
void Hamiltonian::chebyshev_2(
  Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, real bessel_m, int label)
{
  gpu_chebyshev_2<<<grid_size, BLOCK_SIZE>>>(
    n,
    energy_max,
    neighbor_number,
    neighbor_list,
    potential,
    hopping_real,
    hopping_imag,
    state_0.real_part,
    state_0.imag_part,
    state_1.real_part,
    state_1.imag_part,
    state_2.real_part,
    state_2.imag_part,
    state.real_part,
    state.imag_part,
    bessel_m,
    label);
  CHECK(cudaGetLastError());
}

// Kernel which calculates the two first terms of commutator [X, U(dt)]
// Corresponds to Eq. (37) in [Comput. Phys. Commun.185, 28 (2014)].
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

// Wrapper for kernel above
void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, real bessel_1)
{
  gpu_chebyshev_1x<<<grid_size, BLOCK_SIZE>>>(
    n, input.real_part, input.imag_part, output.real_part, output.imag_part, bessel_1);
  CHECK(cudaGetLastError());
}

// Kernel which calculates the further terms of [X, U(dt)]
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
  gpu_chebyshev_2x<<<grid_size, BLOCK_SIZE>>>(
    n,
    energy_max,
    neighbor_number,
    neighbor_list,
    potential,
    hopping_real,
    hopping_imag,
    xx,
    state_0.real_part,
    state_0.imag_part,
    state_0x.real_part,
    state_0x.imag_part,
    state_1.real_part,
    state_1.imag_part,
    state_1x.real_part,
    state_1x.imag_part,
    state_2.real_part,
    state_2.imag_part,
    state_2x.real_part,
    state_2x.imag_part,
    state.real_part,
    state.imag_part,
    bessel_m,
    label);
  CHECK(cudaGetLastError());
}

// Kernel for doing the Chebyshev iteration phi_2 = 2 * H * phi_1 - phi_0.
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

// Wrapper for the Chebyshev iteration
void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
  gpu_kernel_polynomial<<<grid_size, BLOCK_SIZE>>>(
    n,
    energy_max,
    neighbor_number,
    neighbor_list,
    potential,
    hopping_real,
    hopping_imag,
    state_0.real_part,
    state_0.imag_part,
    state_1.real_part,
    state_1.imag_part,
    state_2.real_part,
    state_2.imag_part);
  CHECK(cudaGetLastError());
}
