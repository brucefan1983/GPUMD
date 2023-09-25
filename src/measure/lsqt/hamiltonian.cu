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

#include "force/neighbor.cuh"
#include "hamiltonian.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "vector.cuh"

namespace
{
const int max_neighbor = 50; // do do exceed this

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

__device__ int find_neighbor_cell(
  int cell_id,
  int cell_id_x,
  int cell_id_y,
  int cell_id_z,
  int nx,
  int ny,
  int nz,
  int xx,
  int yy,
  int zz)
{
  int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
  if (cell_id_x + xx < 0)
    neighbor_cell += nx;
  if (cell_id_x + xx >= nx)
    neighbor_cell -= nx;
  if (cell_id_y + yy < 0)
    neighbor_cell += ny * nx;
  if (cell_id_y + yy >= ny)
    neighbor_cell -= ny * nx;
  if (cell_id_z + zz < 0)
    neighbor_cell += nz * ny * nx;
  if (cell_id_z + zz >= nz)
    neighbor_cell -= nz * ny * nx;

  return neighbor_cell;
}

__global__ void construct_hamiltonian(
  const float rc,
  const int N,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_cell_count,
  const int* g_cell_count_sum,
  const int* g_cell_contents,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  int* g_neighbor_number,
  int* g_neighbor_list,
  real* g_potential,
  real* g_hopping_real,
  real* g_hopping_imag,
  real* g_xx)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(box, x1, y1, z1, 2.0f / rc, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;
  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell =
          find_neighbor_cell(cell_id, cell_id_x, cell_id_y, cell_id_z, nx, ny, nz, xx, yy, zz);
        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];
        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];
          if (n1 == n2) {
            continue;
          }
          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float r12[3] = {float(x12double), float(y12double), float(z12double)};
          float d12_2 = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
          if (d12_2 < rc * rc) {
            float d12 = sqrt(d12_2);
          }
        }
      }
    }
  }
}

} // namespace

void Hamiltonian::initialize(real emax, Atom& atom, Box& box)
{
  number_of_atoms = atom.number_of_atoms;
  energy_max = emax;
  tb_cutoff = 2.1; // TODO
  cell_count.resize(number_of_atoms);
  cell_count_sum.resize(number_of_atoms);
  cell_contents.resize(number_of_atoms);
  neighbor_number.resize(number_of_atoms);
  neighbor_list.resize(number_of_atoms * max_neighbor);
  potential.resize(number_of_atoms);
  hopping_real.resize(number_of_atoms * max_neighbor);
  hopping_imag.resize(number_of_atoms * max_neighbor);
  xx.resize(number_of_atoms * max_neighbor);

  int num_bins[3];
  box.get_num_bins(0.5 * tb_cutoff, num_bins);

  find_cell_list(
    0.5 * tb_cutoff,
    num_bins,
    box,
    atom.position_per_atom,
    cell_count,
    cell_count_sum,
    cell_contents);

  construct_hamiltonian<<<(number_of_atoms - 1) / 64 + 1, 64>>>(
    tb_cutoff,
    number_of_atoms,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    box,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2,
    neighbor_number.data(),
    neighbor_list.data(),
    potential.data(),
    hopping_real.data(),
    hopping_imag.data(),
    xx.data());
}

// |output> = H |input>
void Hamiltonian::apply(Vector& input, Vector& output)
{
  gpu_apply_hamiltonian<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    energy_max,
    neighbor_number.data(),
    neighbor_list.data(),
    potential.data(),
    hopping_real.data(),
    hopping_imag.data(),
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part);
  CHECK(cudaGetLastError());
}

// |output> = [X, H] |input>
void Hamiltonian::apply_commutator(Vector& input, Vector& output)
{
  gpu_apply_commutator<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    energy_max,
    neighbor_number.data(),
    neighbor_list.data(),
    hopping_real.data(),
    hopping_imag.data(),
    xx.data(),
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part);
  CHECK(cudaGetLastError());
}

// |output> = V |input>
void Hamiltonian::apply_current(Vector& input, Vector& output)
{
  gpu_apply_current<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    neighbor_number.data(),
    neighbor_list.data(),
    hopping_real.data(),
    hopping_imag.data(),
    xx.data(),
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part);
  CHECK(cudaGetLastError());
}

// Wrapper for the kernel above
void Hamiltonian::chebyshev_01(
  Vector& state_0, Vector& state_1, Vector& state, real bessel_0, real bessel_1, int direction)
{
  gpu_chebyshev_01<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
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

// Wrapper for the kernel above
void Hamiltonian::chebyshev_2(
  Vector& state_0, Vector& state_1, Vector& state_2, Vector& state, real bessel_m, int label)
{
  gpu_chebyshev_2<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    energy_max,
    neighbor_number.data(),
    neighbor_list.data(),
    potential.data(),
    hopping_real.data(),
    hopping_imag.data(),
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

void Hamiltonian::chebyshev_1x(Vector& input, Vector& output, real bessel_1)
{
  gpu_chebyshev_1x<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    input.real_part,
    input.imag_part,
    output.real_part,
    output.imag_part,
    bessel_1);
  CHECK(cudaGetLastError());
}

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
  gpu_chebyshev_2x<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    energy_max,
    neighbor_number.data(),
    neighbor_list.data(),
    potential.data(),
    hopping_real.data(),
    hopping_imag.data(),
    xx.data(),
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

void Hamiltonian::kernel_polynomial(Vector& state_0, Vector& state_1, Vector& state_2)
{
  gpu_kernel_polynomial<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    energy_max,
    neighbor_number.data(),
    neighbor_list.data(),
    potential.data(),
    hopping_real.data(),
    hopping_imag.data(),
    state_0.real_part,
    state_0.imag_part,
    state_1.real_part,
    state_1.imag_part,
    state_2.real_part,
    state_2.imag_part);
  CHECK(cudaGetLastError());
}
