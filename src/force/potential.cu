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

/*----------------------------------------------------------------------------80
The abstract base class (ABC) for the potential classes.
------------------------------------------------------------------------------*/

#include "potential.cuh"
#include "utilities/error.cuh"
#define BLOCK_SIZE_FORCE 64
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

Potential::Potential(void) { rc = 0.0; }

Potential::~Potential(void)
{
  // nothing
}

static __global__ void gpu_find_force_many_body(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const double* __restrict__ g_f12x,
  const double* __restrict__ g_f12y,
  const double* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  double s_fx = 0.0;  // force_x
  double s_fy = 0.0;  // force_y
  double s_fz = 0.0;  // force_z
  double s_sxx = 0.0; // virial_stress_xx
  double s_sxy = 0.0; // virial_stress_xy
  double s_sxz = 0.0; // virial_stress_xz
  double s_syx = 0.0; // virial_stress_yx
  double s_syy = 0.0; // virial_stress_yy
  double s_syz = 0.0; // virial_stress_yz
  double s_szx = 0.0; // virial_stress_zx
  double s_szy = 0.0; // virial_stress_zy
  double s_szz = 0.0; // virial_stress_zz

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];

      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);

      double f12x = g_f12x[index];
      double f12y = g_f12y[index];
      double f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = offset * number_of_particles + n2;
      double f21x = g_f12x[index];
      double f21y = g_f12y[index];
      double f21z = g_f12z[index];

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
      s_sxx += x12 * f21x;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syy += y12 * f21y;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
      s_szz += z12 * f21z;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}

void Potential::find_properties_many_body(
  Box& box,
  const int* NN,
  const int* NL,
  const double* f12x,
  const double* f12y,
  const double* f12z,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = position_per_atom.size() / 3;
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

  gpu_find_force_many_body<<<grid_size, BLOCK_SIZE_FORCE>>>(
    number_of_atoms, N1, N2, box, NN, NL, f12x, f12y, f12z, position_per_atom.data(),
    position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
    force_per_atom.data(), force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms, virial_per_atom.data());
  CUDA_CHECK_KERNEL
}

static __global__ void gpu_find_neighbor_ON2(
  const Box box,
  const int N,
  const int N1,
  const int N2,
  const double rc2,
  const double* x,
  const double* y,
  const double* z,
  int* NN,
  int* NL)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;

  if (n1 < N2) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int count = 0;

    for (int n2 = N1; n2 < N2; ++n2) {
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double z12 = z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

      if (n1 != n2 && d2 < rc2) {
        NL[count++ * N + n1] = n2;
      }
    }
    NN[n1] = count;
  }
}

static __device__ void find_cell_id(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  if (box.triclinic == 0) {
    cell_id_x = floor(x * rc_inv);
    cell_id_y = floor(y * rc_inv);
    cell_id_z = floor(z * rc_inv);
  } else {
    const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
    const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
    const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
    cell_id_x = floor(sx * box.thickness_x * rc_inv);
    cell_id_y = floor(sy * box.thickness_y * rc_inv);
    cell_id_z = floor(sz * box.thickness_z * rc_inv);
  }
  while (cell_id_x < 0)
    cell_id_x += nx;
  while (cell_id_x >= nx)
    cell_id_x -= nx;
  while (cell_id_y < 0)
    cell_id_y += ny;
  while (cell_id_y >= ny)
    cell_id_y -= ny;
  while (cell_id_z < 0)
    cell_id_z += nz;
  while (cell_id_z >= nz)
    cell_id_z -= nz;
  cell_id = cell_id_x + nx * cell_id_y + nx * ny * cell_id_z;
}

static __device__ void find_cell_id(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(box, x, y, z, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);
}

static __global__ void find_cell_counts(
  const Box box,
  const int N,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents(
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

static __global__ void gpu_find_neighbor_ON1(
  const Box box,
  const int N,
  const int* __restrict__ cell_counts,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  int* NN,
  int* NL,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv,
  const double cutoff_square)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  if (n1 < N) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    const double z1 = z[n1];
    int cell_id;
    int cell_id_x;
    int cell_id_y;
    int cell_id_z;
    find_cell_id(box, x1, y1, z1, rc_inv, nx, ny, nz, cell_id_x, cell_id_y, cell_id_z, cell_id);
    const int klim = box.pbc_z ? 1 : 0;
    const int jlim = box.pbc_y ? 1 : 0;
    const int ilim = box.pbc_x ? 1 : 0;

    for (int k = -klim; k < klim + 1; ++k) {
      for (int j = -jlim; j < jlim + 1; ++j) {
        for (int i = -ilim; i < ilim + 1; ++i) {
          int neighbor_cell = cell_id + k * nx * ny + j * nx + i;
          if (cell_id_x + i < 0)
            neighbor_cell += nx;
          if (cell_id_x + i >= nx)
            neighbor_cell -= nx;
          if (cell_id_y + j < 0)
            neighbor_cell += ny * nx;
          if (cell_id_y + j >= ny)
            neighbor_cell -= ny * nx;
          if (cell_id_z + k < 0)
            neighbor_cell += nz * ny * nx;
          if (cell_id_z + k >= nz)
            neighbor_cell -= nz * ny * nx;

          const int num_atoms_neighbor_cell = cell_counts[neighbor_cell];
          const int num_atoms_previous_cells = cell_count_sum[neighbor_cell];

          for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
            const int n2 = cell_contents[num_atoms_previous_cells + m];
            double x12 = x[n2] - x1;
            double y12 = y[n2] - y1;
            double z12 = z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            const double d2 = x12 * x12 + y12 * y12 + z12 * z12;

            if (n1 != n2 && d2 < cutoff_square) {
              NL[count * N + n1] = n2;
              count++;
            }
          }
        }
      }
    }
    NN[n1] = count;
  }
}

#ifdef DEBUG
static __global__ void gpu_sort_neighbor_list(const int N, const int* NN, int* NL)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int neighbor_number = NN[bid];
  int atom_index;
  extern __shared__ int atom_index_copy[];

  if (tid < neighbor_number) {
    atom_index = NL[bid + tid * N];
    atom_index_copy[tid] = atom_index;
  }
  int count = 0;
  __syncthreads();

  for (int j = 0; j < neighbor_number; ++j) {
    if (atom_index > atom_index_copy[j]) {
      count++;
    }
  }

  if (tid < neighbor_number) {
    NL[bid + count * N] = atom_index;
  }
}
#endif

void Potential::find_neighbor(
  Box& box, const GPU_Vector<double>& position_per_atom, GPU_Vector<int>& NN, GPU_Vector<int>& NL)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N2 - N1 - 1) / block_size + 1;
  const double rc2 = rc * rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + N;
  const double* z = position_per_atom.data() + N * 2;

  int num_bins[3];
  const bool use_ON2 = box.get_num_bins(rc, num_bins);

  if (use_ON2) {
    gpu_find_neighbor_ON2<<<grid_size, block_size>>>(
      box, N, N1, N2, rc2, x, y, z, NN.data(), NL.data());
    CUDA_CHECK_KERNEL
  } else {
    const double rc_inv = 1.0 / rc;
    const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

    CHECK(cudaMemset(cell_count.data(), 0, sizeof(int) * N_cells));
    CHECK(cudaMemset(cell_count_sum.data(), 0, sizeof(int) * N_cells));
    CHECK(cudaMemset(cell_contents.data(), 0, sizeof(int) * N));

    find_cell_counts<<<grid_size, block_size>>>(
      box, N, cell_count.data(), x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv);
    CUDA_CHECK_KERNEL

    thrust::exclusive_scan(
      thrust::device, cell_count.data(), cell_count.data() + N_cells, cell_count_sum.data());

    CHECK(cudaMemset(cell_count.data(), 0, sizeof(int) * N_cells));

    find_cell_contents<<<grid_size, block_size>>>(
      box, N, cell_count.data(), cell_count_sum.data(), cell_contents.data(), x, y, z, num_bins[0],
      num_bins[1], num_bins[2], rc_inv);
    CUDA_CHECK_KERNEL

    gpu_find_neighbor_ON1<<<grid_size, block_size>>>(
      box, N, cell_count.data(), cell_count_sum.data(), cell_contents.data(), NN.data(), NL.data(),
      x, y, z, num_bins[0], num_bins[1], num_bins[2], rc_inv, rc2);
    CUDA_CHECK_KERNEL
#ifdef DEBUG
    const int MN = NL.size() / NN.size();
    const int smem = MN * sizeof(int);
    gpu_sort_neighbor_list<<<N, MN, smem>>>(N, NN.data(), NL.data());
#endif
  }
}
