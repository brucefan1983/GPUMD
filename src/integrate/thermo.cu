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
Calculate global thermodynamic properties from basic per-atom data.
------------------------------------------------------------------------------*/

#include "thermo.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#define DIM 3

// g_thermo[0-7] = T, U, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz
static __global__ void gpu_find_thermo_single(
  const int N,
  const int N_temperature,
  const double volume,
  const double* g_mass,
  const double* g_potential,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_sxx,
  const double* g_syy,
  const double* g_szz,
  const double* g_sxy,
  const double* g_sxz,
  const double* g_syz,
  double* g_thermo)
{
  const int tid = threadIdx.x;
  const int quantity = blockIdx.x;
  const int number_of_batches = (N - 1) / blockDim.x + 1;
  double mass, vx, vy, vz;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  switch (quantity) {
    case 0:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vx = g_vx[n];
          vy = g_vy[n];
          vz = g_vz[n];
          s_data[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
      }
      break;
    case 1:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          s_data[tid] += g_potential[n];
        }
      }
      break;
    case 2:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vx = g_vx[n];
          s_data[tid] += g_sxx[n] + vx * vx * mass;
        }
      }
      break;
    case 3:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vy = g_vy[n];
          s_data[tid] += g_syy[n] + vy * vy * mass;
        }
      }
      break;
    case 4:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vz = g_vz[n];
          s_data[tid] += g_szz[n] + vz * vz * mass;
        }
      }
      break;
    case 5:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vx = g_vx[n];
          vy = g_vy[n];
          s_data[tid] += g_sxy[n] + vx * vy * mass;
        }
      }
      break;
    case 6:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vx = g_vx[n];
          vz = g_vz[n];
          s_data[tid] += g_sxz[n] + vx * vz * mass;
        }
      }
      break;
    case 7:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        const int n = tid + batch * blockDim.x;
        if (n < N) {
          mass = g_mass[n];
          vy = g_vy[n];
          vz = g_vz[n];
          s_data[tid] += g_syz[n] + vy * vz * mass;
        }
      }
      break;
  }

  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (quantity == 0) {
      g_thermo[0] = s_data[0] / (DIM * N_temperature * K_B);
    } else if (quantity == 1) {
      g_thermo[1] = s_data[0];
    } else {
      g_thermo[quantity] = s_data[0] / volume;
    }
  }
}

// First stage for large systems. Each block computes one partial sum for one
// thermodynamic quantity.
static __global__ void gpu_find_thermo_partial(
  const int N,
  const double* g_mass,
  const double* g_potential,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_sxx,
  const double* g_syy,
  const double* g_szz,
  const double* g_sxy,
  const double* g_sxz,
  const double* g_syz,
  double* g_partial)
{
  const int tid = threadIdx.x;
  const int quantity = blockIdx.y;
  const int partial_index = blockIdx.x;
  const int stride = gridDim.x * blockDim.x;
  double sum = 0.0;
  double mass, vx, vy, vz;

  switch (quantity) {
    case 0:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vx = g_vx[n];
        vy = g_vy[n];
        vz = g_vz[n];
        sum += (vx * vx + vy * vy + vz * vz) * mass;
      }
      break;
    case 1:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        sum += g_potential[n];
      }
      break;
    case 2:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vx = g_vx[n];
        sum += g_sxx[n] + vx * vx * mass;
      }
      break;
    case 3:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vy = g_vy[n];
        sum += g_syy[n] + vy * vy * mass;
      }
      break;
    case 4:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vz = g_vz[n];
        sum += g_szz[n] + vz * vz * mass;
      }
      break;
    case 5:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vx = g_vx[n];
        vy = g_vy[n];
        sum += g_sxy[n] + vx * vy * mass;
      }
      break;
    case 6:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vx = g_vx[n];
        vz = g_vz[n];
        sum += g_sxz[n] + vx * vz * mass;
      }
      break;
    case 7:
      for (int n = partial_index * blockDim.x + tid; n < N; n += stride) {
        mass = g_mass[n];
        vy = g_vy[n];
        vz = g_vz[n];
        sum += g_syz[n] + vy * vz * mass;
      }
      break;
  }

  __shared__ double s_data[256];
  s_data[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_partial[quantity * gridDim.x + partial_index] = s_data[0];
  }
}

// Second stage for large systems. The reduction order is fixed and no atomics
// are used, so repeated runs of the same executable remain deterministic.
static __global__ void gpu_reduce_thermo_partial(
  const int number_of_partial_blocks,
  const int N_temperature,
  const double volume,
  const double* g_partial,
  double* g_thermo)
{
  const int tid = threadIdx.x;
  const int quantity = blockIdx.x;
  double sum = 0.0;

  for (int n = tid; n < number_of_partial_blocks; n += blockDim.x) {
    sum += g_partial[quantity * number_of_partial_blocks + n];
  }

  __shared__ double s_data[1024];
  s_data[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (quantity == 0) {
      g_thermo[0] = s_data[0] / (DIM * N_temperature * K_B);
    } else if (quantity == 1) {
      g_thermo[1] = s_data[0];
    } else {
      g_thermo[quantity] = s_data[0] / volume;
    }
  }
}

void Thermo::compute(
  const int number_of_atoms_for_temperature,
  const double volume,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& thermo)
{
  const int number_of_atoms = mass.size();

  // Use the smallest power-of-two block that covers small systems.
  int block_size = 32;
  while (block_size < number_of_atoms && block_size < 1024) {
    block_size <<= 1;
  }

  // Use a two-stage reduction once 16 or more 256-thread partial blocks are
  // available. Smaller systems avoid the extra kernel launch.
  int number_of_partial_blocks = (number_of_atoms + 255) / 256;
  if (number_of_partial_blocks < 16) {
    gpu_find_thermo_single<<<8, block_size>>>(
      number_of_atoms,
      number_of_atoms_for_temperature,
      volume,
      mass.data(),
      potential_per_atom.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms,
      velocity_per_atom.data() + 2 * number_of_atoms,
      virial_per_atom.data(),
      virial_per_atom.data() + number_of_atoms,
      virial_per_atom.data() + number_of_atoms * 2,
      virial_per_atom.data() + number_of_atoms * 3,
      virial_per_atom.data() + number_of_atoms * 4,
      virial_per_atom.data() + number_of_atoms * 5,
      thermo.data());
    GPU_CHECK_KERNEL
    return;
  }

  if (number_of_sms_ == 0) {
    gpuDeviceProp device_prop;
    CHECK(gpuGetDeviceProperties(&device_prop, 0));
    number_of_sms_ = device_prop.multiProcessorCount;
  }

  const int max_partial_blocks = number_of_sms_ * 2 < 1024 ? number_of_sms_ * 2 : 1024;
  if (number_of_partial_blocks > max_partial_blocks) {
    number_of_partial_blocks = max_partial_blocks;
  }

  const int partial_size = 8 * number_of_partial_blocks;
  if (partial_.size() < partial_size) {
    partial_.resize(partial_size);
  }

  const dim3 grid(number_of_partial_blocks, 8);
  gpu_find_thermo_partial<<<grid, 256>>>(
    number_of_atoms,
    mass.data(),
    potential_per_atom.data(),
    velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms,
    virial_per_atom.data(),
    virial_per_atom.data() + number_of_atoms,
    virial_per_atom.data() + number_of_atoms * 2,
    virial_per_atom.data() + number_of_atoms * 3,
    virial_per_atom.data() + number_of_atoms * 4,
    virial_per_atom.data() + number_of_atoms * 5,
    partial_.data());
  GPU_CHECK_KERNEL

  int reduction_block_size = 32;
  while (reduction_block_size < number_of_partial_blocks && reduction_block_size < 1024) {
    reduction_block_size <<= 1;
  }
  gpu_reduce_thermo_partial<<<8, reduction_block_size>>>(
    number_of_partial_blocks,
    number_of_atoms_for_temperature,
    volume,
    partial_.data(),
    thermo.data());
  GPU_CHECK_KERNEL
}
