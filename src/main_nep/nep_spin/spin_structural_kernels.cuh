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

/*----------------------------------------------------------------------------80
Major contributors:
  Chengbing Chen: Initial implementation.
------------------------------------------------------------------------------*/

#pragma once

namespace
{
constexpr int structural_radial_lanes = 8;
constexpr int structural_radial_block_size = 128;

__device__ __forceinline__ unsigned structural_radial_mask()
{
  return 0xffu << (threadIdx.x % 32 / structural_radial_lanes * structural_radial_lanes);
}

__device__ __forceinline__ void
structural_radial_basis(int center_type, int neighbor_type, float x, float y, float z, float* basis)
{
  const float distance = sqrtf(x * x + y * y + z * z);
  const float cutoff = get_rc_radial_jit(center_type, neighbor_type);
  const float inverse_cutoff = 1.0f / cutoff;
  float cutoff_value;
  find_fc(cutoff, inverse_cutoff, distance, cutoff_value);
  find_fn(BASIS_SIZE_RADIAL_JIT, inverse_cutoff, distance, cutoff_value, basis);
}

__global__ void spin_structural_radial_descriptor(
  int count,
  const int* offsets,
  const int* neighbors,
  const int* neighbor_index,
  const int* type,
  const float* x,
  const float* y,
  const float* z,
  const float* parameters,
  float* descriptors)
{
  const int atom = (blockIdx.x * blockDim.x + threadIdx.x) / structural_radial_lanes;
  if (atom >= count) {
    return;
  }
  const int lane = threadIdx.x % structural_radial_lanes;
  const unsigned mask = structural_radial_mask();
  const int center_type = NUM_TYPES_JIT == 1 ? 0 : type[atom];
  const float* coefficients = get_c_jit(parameters);
  const int offset = offsets[atom];
  const int length = neighbors[atom];
  float descriptor[DIM_RADIAL_JIT] = {};
  constexpr int basis_count = BASIS_SIZE_RADIAL_JIT + 1;
  constexpr int basis_storage = basis_count < 2 ? 2 : basis_count;

  for (int first = 0; first < length; first += structural_radial_lanes) {
    float contribution[DIM_RADIAL_JIT] = {};
    if (first + lane < length) {
      const int edge = offset + first + lane;
      const int neighbor_type = NUM_TYPES_JIT == 1 ? 0 : type[neighbor_index[edge]];
      const int pair = descriptor_type_index_jit(center_type, neighbor_type);
      float basis[basis_storage];
      structural_radial_basis(center_type, neighbor_type, x[edge], y[edge], z[edge], basis);
#pragma unroll
      for (int n = 0; n < DIM_RADIAL_JIT; ++n) {
#pragma unroll
        for (int k = 0; k < basis_count; ++k) {
          const int index = get_c_index(pair, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
          contribution[n] += basis[k] * coefficients[index];
        }
      }
    }
    // Preserve CSR neighbor order in the direct contraction.
    for (int source = 0; source < structural_radial_lanes && first + source < length; ++source) {
#pragma unroll
      for (int n = 0; n < DIM_RADIAL_JIT; ++n) {
        descriptor[n] += __shfl_sync(mask, contribution[n], source, structural_radial_lanes);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < DIM_RADIAL_JIT; ++n) {
      descriptors[atom + n * count] = descriptor[n];
    }
  }
}

__global__ void spin_structural_radial_force(
  int count,
  const int* offsets,
  const int* neighbors,
  const int* neighbor_index,
  const int* type,
  const float* x,
  const float* y,
  const float* z,
  const float* parameters,
  const float* descriptor_gradient,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int atom = (blockIdx.x * blockDim.x + threadIdx.x) / structural_radial_lanes;
  if (atom >= count) {
    return;
  }
  const int lane = threadIdx.x % structural_radial_lanes;
  const unsigned mask = structural_radial_mask();
  const int center_type = NUM_TYPES_JIT == 1 ? 0 : type[atom];
  const float* coefficients = get_c_jit(parameters);
  const int offset = offsets[atom];
  const int length = neighbors[atom];
  float center_force[3] = {};
  float center_virial[6] = {};
  for (int first = 0; first < length; first += structural_radial_lanes) {
    float displacement[3] = {};
    float edge_force[3] = {};
    if (first + lane < length) {
      const int edge = offset + first + lane;
      const int neighbor = neighbor_index[edge];
      const int neighbor_type = NUM_TYPES_JIT == 1 ? 0 : type[neighbor];
      const int pair = descriptor_type_index_jit(center_type, neighbor_type);
      displacement[0] = x[edge];
      displacement[1] = y[edge];
      displacement[2] = z[edge];
      const float distance = sqrtf(
        displacement[0] * displacement[0] + displacement[1] * displacement[1] +
        displacement[2] * displacement[2]);
      const float inverse_distance = 1.0f / distance;
      const float cutoff = get_rc_radial_jit(center_type, neighbor_type);
      const float inverse_cutoff = 1.0f / cutoff;
      float cutoff_value, cutoff_derivative;
      find_fc_and_fcp(cutoff, inverse_cutoff, distance, cutoff_value, cutoff_derivative);
      constexpr int basis_storage = BASIS_SIZE_RADIAL_JIT == 0 ? 2 : BASIS_SIZE_RADIAL_JIT + 1;
      float basis[basis_storage], derivative[basis_storage];
      find_fn_and_fnp(
        BASIS_SIZE_RADIAL_JIT,
        inverse_cutoff,
        distance,
        cutoff_value,
        cutoff_derivative,
        basis,
        derivative);
#pragma unroll
      for (int n = 0; n < DIM_RADIAL_JIT; ++n) {
        float contracted = 0.0f;
#pragma unroll
        for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
          const int index = get_c_index(pair, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
          contracted += derivative[k] * coefficients[index];
        }
        const float scale = descriptor_gradient[atom + n * count] * (contracted * inverse_distance);
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
          edge_force[axis] += scale * displacement[axis];
        }
      }
      atomicAdd(&fx[neighbor], -edge_force[0]);
      atomicAdd(&fy[neighbor], -edge_force[1]);
      atomicAdd(&fz[neighbor], -edge_force[2]);
    }

    for (int source = 0; source < structural_radial_lanes && first + source < length; ++source) {
      float r[3], f[3];
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        r[axis] = __shfl_sync(mask, displacement[axis], source, structural_radial_lanes);
        f[axis] = __shfl_sync(mask, edge_force[axis], source, structural_radial_lanes);
        center_force[axis] += f[axis];
      }
      center_virial[0] -= r[0] * f[0];
      center_virial[1] -= r[1] * f[1];
      center_virial[2] -= r[2] * f[2];
      center_virial[3] -= r[0] * f[1];
      center_virial[4] -= r[1] * f[2];
      center_virial[5] -= r[2] * f[0];
    }
  }

  if (lane != 0) {
    return;
  }

  atomicAdd(&fx[atom], center_force[0]);
  atomicAdd(&fy[atom], center_force[1]);
  atomicAdd(&fz[atom], center_force[2]);
#pragma unroll
  for (int axis = 0; axis < 3; ++axis) {
    virial[atom + axis * count] += center_virial[axis];
    // The radial kernel initializes the Spin structural shear buffer.
    virial[atom + (axis + 3) * count] = center_virial[axis + 3];
  }
}

static __global__ void spin_structural_angular_descriptor(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N)
    return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_ANGULAR_JIT] = {0.0f};

#pragma unroll
  // Share edge geometry and radial basis across five angular channels.
  for (int base = 0; base <= N_MAX_ANGULAR_JIT; base += 5) {
    float sums[5][NUM_ABC_JIT] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = NN_sum[n1] + i1;
      const int n2 = NL[index];
      const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
      const float x12 = x12_all[index];
      const float y12 = y12_all[index];
      const float z12 = z12_all[index];
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      const float rc = get_rc_angular_jit(t1, t2);
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      constexpr int basis_storage = BASIS_SIZE_ANGULAR_JIT == 0 ? 2 : BASIS_SIZE_ANGULAR_JIT + 1;
      float fn12[basis_storage];
      find_fn(BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fn12);

      const int type_index = descriptor_type_index_jit(t1, t2);
#pragma unroll
      for (int lane = 0; lane < 5; ++lane) {
        const int n = base + lane;
        if (n > N_MAX_ANGULAR_JIT)
          continue;
        float gn12 = 0.0f;
#pragma unroll
        for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
          const int c_index = get_c_index(
            type_index, n, k, N_MAX_ANGULAR_JIT, BASIS_SIZE_ANGULAR_JIT, NUM_C_RADIAL_JIT);
          gn12 += fn12[k] * c[c_index];
        }
        accumulate_s(L_MAX_JIT, d12, x12, y12, z12, gn12, sums[lane]);
      }
    }

#pragma unroll
    for (int lane = 0; lane < 5; ++lane) {
      const int n = base + lane;
      if (n > N_MAX_ANGULAR_JIT)
        continue;
      float* s = sums[lane];
      find_q(
        L_MAX_JIT,
        HAS_Q_222_JIT,
        HAS_Q_1111_JIT,
        HAS_Q_112_JIT,
        HAS_Q_123_JIT,
        HAS_Q_233_JIT,
        HAS_Q_134_JIT,
        N_MAX_ANGULAR_JIT + 1,
        n,
        s,
        q);

#pragma unroll
      for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
        sum_fxyz[(n * NUM_ABC_JIT + abc) * N + n1] = s[abc];
      }
    }
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int l = 0; l < NUM_L_JIT; ++l) {
      const int ln = l * (N_MAX_ANGULAR_JIT + 1) + n;
      descriptors[n1 + (DIM_RADIAL_JIT + ln) * N] = q[ln];
    }
  }
}

} // namespace

extern "C" int nep_train_launch_descriptor_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  spin_structural_angular_descriptor<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors, sum_fxyz);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_descriptor_radial(
  int count,
  const int* offsets,
  const int* neighbors,
  const int* neighbor_index,
  const int* type,
  const float* x,
  const float* y,
  const float* z,
  const float* parameters,
  float* descriptors)
{
  constexpr int atoms_per_block = structural_radial_block_size / structural_radial_lanes;
  const int blocks = (count + atoms_per_block - 1) / atoms_per_block;
  spin_structural_radial_descriptor<<<blocks, structural_radial_block_size>>>(
    count, offsets, neighbors, neighbor_index, type, x, y, z, parameters, descriptors);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_radial(
  int count,
  const int* offsets,
  const int* neighbors,
  const int* neighbor_index,
  const int* type,
  const float* x,
  const float* y,
  const float* z,
  const float* parameters,
  const float* descriptor_gradient,
  const float*,
  const float*,
  const float*,
  const float*,
  int,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  constexpr int atoms_per_block = structural_radial_block_size / structural_radial_lanes;
  const int blocks = (count + atoms_per_block - 1) / atoms_per_block;
  spin_structural_radial_force<<<blocks, structural_radial_block_size>>>(
    count,
    offsets,
    neighbors,
    neighbor_index,
    type,
    x,
    y,
    z,
    parameters,
    descriptor_gradient,
    fx,
    fy,
    fz,
    virial);
  return static_cast<int>(cudaGetLastError());
}
