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
Construct the neighbor list, choosing the O(N) or O(N^2) method automatically
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

// determining whether a new neighbor list should be built
static __global__ void gpu_check_atom_distance(
  int N,
  double d2,
  double* x_old,
  double* y_old,
  double* z_old,
  double* x_new,
  double* y_new,
  double* z_new,
  int* g_sum)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int n = bid * blockDim.x + tid;
  __shared__ int s_sum[128];
  s_sum[tid] = 0;
  if (n < N) {
    double dx = x_new[n] - x_old[n];
    double dy = y_new[n] - y_old[n];
    double dz = z_new[n] - z_old[n];
    if ((dx * dx + dy * dy + dz * dz) > d2) {
      s_sum[tid] = 1;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(g_sum, s_sum[0]);
  }
}

__device__ int static_s2[1];

// If the returned value > 0, the neighbor list will be updated.
int Neighbor::check_atom_distance(double* x, double* y, double* z)
{
  const int N = NN.size();
  double d2 = skin * skin * 0.25;
  int* gpu_s2;
  CHECK(cudaGetSymbolAddress((void**)&gpu_s2, static_s2));
  int cpu_s2[1] = {0};
  CHECK(cudaMemcpy(gpu_s2, cpu_s2, sizeof(int), cudaMemcpyHostToDevice));
  gpu_check_atom_distance<<<(N - 1) / 128 + 1, 128>>>(
    N, d2, x0.data(), y0.data(), z0.data(), x, y, z, gpu_s2);
  CUDA_CHECK_KERNEL
  CHECK(cudaMemcpy(cpu_s2, gpu_s2, sizeof(int), cudaMemcpyDeviceToHost));
  return cpu_s2[0];
}

// pull the atoms back to the box after updating the neighbor list
static __global__ void gpu_apply_pbc(int N, Box box, double* g_x, double* g_y, double* g_z)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    if (box.triclinic == 0) {
      double lx = box.cpu_h[0];
      double ly = box.cpu_h[1];
      double lz = box.cpu_h[2];
      if (box.pbc_x == 1) {
        if (g_x[n] < 0) {
          g_x[n] += lx;
        } else if (g_x[n] > lx) {
          g_x[n] -= lx;
        }
      }
      if (box.pbc_y == 1) {
        if (g_y[n] < 0) {
          g_y[n] += ly;
        } else if (g_y[n] > ly) {
          g_y[n] -= ly;
        }
      }
      if (box.pbc_z == 1) {
        if (g_z[n] < 0) {
          g_z[n] += lz;
        } else if (g_z[n] > lz) {
          g_z[n] -= lz;
        }
      }
    } else {
      double x = g_x[n];
      double y = g_y[n];
      double z = g_z[n];
      double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
      double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
      double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
      if (box.pbc_x == 1) {
        if (sx < 0.0) {
          sx += 1.0;
        } else if (sx > 1.0) {
          sx -= 1.0;
        }
      }
      if (box.pbc_y == 1) {
        if (sy < 0.0) {
          sy += 1.0;
        } else if (sy > 1.0) {
          sy -= 1.0;
        }
      }
      if (box.pbc_z == 1) {
        if (sz < 0.0) {
          sz += 1.0;
        } else if (sz > 1.0) {
          sz -= 1.0;
        }
      }
      g_x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
      g_y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
      g_z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
    }
  }
}

// update the reference positions:
static __global__ void
gpu_update_xyz0(int N, double* x, double* y, double* z, double* x0, double* y0, double* z0)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    x0[n] = x[n];
    y0[n] = y[n];
    z0[n] = z[n];
  }
}

// check the bound of the neighbor list
void Neighbor::check_bound(const bool is_first)
{
  const int N = NN.size();
  NN.copy_to_host(cpu_NN.data());
  int flag = 0;
  max_NN = 0;
  for (int n = 0; n < N; ++n) {
    if (cpu_NN[n] > max_NN) {
      max_NN = cpu_NN[n];
    }
    if (cpu_NN[n] > MN) {
      printf("Error: NN[%d] = %d > %d\n", n, cpu_NN[n], MN);
      flag = 1;
    }
  }
  if (flag == 1) {
    exit(1);
  } else if (is_first) {
    printf("Build the initial neighbor list with cutoff %g A and size %d.\n", rc, MN);
    printf("    calculated maximum number of neighbors for one atom in the system = %d\n", max_NN);
  }
}

// simple version for sorting the neighbor indicies of each atom
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

void Neighbor::find_neighbor(Box& box, double* x, double* y, double* z)
{
  const int N = NN.size();
  int num_bins[3];
  bool use_ON2 = box.get_num_bins(rc, num_bins);

  if (use_ON2) {
    find_neighbor_ON2(box, x, y, z);
  } else {
    find_neighbor_ON1(num_bins[0], num_bins[1], num_bins[2], box, x, y, z);
#ifdef DEBUG
    if (MN > 1024) {
      PRINT_INPUT_ERROR("MN > 1024\n");
    }
    const int smem = MN * sizeof(int);
    gpu_sort_neighbor_list<<<N, MN, smem>>>(N, NN.data(), NL.data());
#endif
  }
}

// the driver function to be called outside this file
void Neighbor::find_neighbor(
  const bool is_first, Box& box, GPU_Vector<double>& position_per_atom, const double force_rc_max)
{
  const int N = NN.size();
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;

  double* x = position_per_atom.data();
  double* y = position_per_atom.data() + N;
  double* z = position_per_atom.data() + N * 2;

  if (is_first) {

    find_neighbor(box, x, y, z);
    check_bound(is_first);

    gpu_update_xyz0<<<grid_size, block_size>>>(N, x, y, z, x0.data(), y0.data(), z0.data());
    CUDA_CHECK_KERNEL
  } else {
    rc = force_rc_max + skin; // update rc when we know the largest force cutoff

    int update = check_atom_distance(x, y, z);

    if (update) {
      number_of_updates++;

      find_neighbor(box, x, y, z);
      check_bound(is_first);

      gpu_apply_pbc<<<grid_size, block_size>>>(N, box, x, y, z);
      CUDA_CHECK_KERNEL

      gpu_update_xyz0<<<grid_size, block_size>>>(N, x, y, z, x0.data(), y0.data(), z0.data());
      CUDA_CHECK_KERNEL
    }
  }
}

void Neighbor::finalize()
{
  update = 1;
  number_of_updates = 0;
  max_NN = 0;
}
