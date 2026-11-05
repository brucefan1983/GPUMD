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

#include "dataset.cuh"
#include "parameters.cuh"
#include "tb.cuh"

#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
#include <hip/hip_complex.h>
#include <hipsolver/hipsolver.h>
typedef hipFloatComplex tbComplex;
#define tbsolverDnCheevjBatched_bufferSize hipsolverDnCheevjBatched_bufferSize
#define tbsolverDnCheevjBatched hipsolverDnCheevjBatched
#else
#include <cuComplex.h>
#include <cusolverDn.h>
typedef cuComplex tbComplex;
#define tbsolverDnCheevjBatched_bufferSize cusolverDnCheevjBatched_bufferSize
#define tbsolverDnCheevjBatched cusolverDnCheevjBatched
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "utilities/nep_utilities.cuh"

namespace
{

struct DescriptorMeta {
  float rc_radial;
  float rc_angular;
  int n_max_radial;
  int n_max_angular;
  int basis_size_radial;
  int basis_size_angular;
  int L_max;
  int num_c_radial;
  int num_types;
  int descriptor_block_size;
};

DescriptorMeta descriptor_meta(const Parameters& para)
{
  return {
    static_cast<float>(para.rc_radial),
    static_cast<float>(para.rc_angular),
    para.n_max_radial,
    para.n_max_angular,
    para.basis_size_radial,
    para.basis_size_angular,
    para.L_max,
    para.num_c_radial,
    para.num_types,
    para.descriptor_block_size};
}

struct NeighborView {
  const int* nn_sum;
  const int* nn;
  const int* nl;
  const float* dx;
  const float* dy;
  const float* dz;
};

struct OnsiteMeta {
  int dim;
  int hidden;
  int num_outputs;
  int block_size;
  int parameter_offset;
};

OnsiteMeta onsite_meta(const Parameters& para)
{
  return {
    para.dim,
    para.num_neurons_onsite,
    para.num_onsite_outputs,
    para.onsite_block_size,
    para.offset_onsite};
}

struct HoppingMeta {
  int num_types;
  int dim;
  int dim_radial;
  int hidden;
  int homo_outputs;
  int hetero_outputs;
  int max_outputs;
  float rc_radial;
  int basis_size_radial;
  int descriptor_block_size;
  float rc_hopping;
  int parameter_offset;
  int descriptor_offset;
};

HoppingMeta hopping_meta(const Parameters& para)
{
  return {
    para.num_types,
    para.dim,
    para.dim_radial,
    para.num_neurons_hopping,
    para.num_hopping_outputs_homo,
    para.num_hopping_outputs_hetero,
    para.max_hopping_outputs,
    static_cast<float>(para.rc_radial),
    para.basis_size_radial,
    para.descriptor_block_size,
    static_cast<float>(para.rc_hopping),
    para.offset_hopping,
    para.offset_descriptor};
}

static __device__ __forceinline__ void
find_radial_basis(const float rc, const int basis_size, const float r, float* fn)
{
  const float rcinv = 1.0f / rc;
  float fc = 0.0f;
  find_fc(rc, rcinv, r, fc);
  find_fn(basis_size, rcinv, r, fc, fn);
}

static __global__ void find_descriptors_radial(
  const int N,
  const NeighborView neighbors,
  const int* g_atom_types,
  const DescriptorMeta meta,
  const float* __restrict__ g_c,
  float* g_descriptors)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }

  float q[MAX_NUM_N] = {0.0f};
  for (int m = 0; m < neighbors.nn[i]; ++m) {
    const int index = neighbors.nn_sum[i] + m;
    const int j = neighbors.nl[index];
    const int ti = g_atom_types[i];
    const int tj = g_atom_types[j];
    const int pair_block = (ti * meta.num_types + tj) * meta.descriptor_block_size;
    const float x = neighbors.dx[index];
    const float y = neighbors.dy[index];
    const float z = neighbors.dz[index];
    const float r = sqrtf(x * x + y * y + z * z);

    float fn[MAX_NUM_N];
    find_radial_basis(meta.rc_radial, meta.basis_size_radial, r, fn);

    for (int n = 0; n <= meta.n_max_radial; ++n) {
      float gn = 0.0f;
      const int c0 = pair_block + n * (meta.basis_size_radial + 1);
      for (int k = 0; k <= meta.basis_size_radial; ++k) {
        gn += fn[k] * g_c[c0 + k];
      }
      q[n] += gn;
    }
  }

  for (int n = 0; n <= meta.n_max_radial; ++n) {
    g_descriptors[n * N + i] = q[n];
  }
}

static __device__ __forceinline__ void find_angular_moments(
  const int i,
  const int n,
  const DescriptorMeta& meta,
  const NeighborView neighbors,
  const int* g_atom_types,
  const float* g_c,
  float* s)
{
  for (int m = 0; m < neighbors.nn[i]; ++m) {
    const int index = neighbors.nn_sum[i] + m;
    const int j = neighbors.nl[index];
    const int ti = g_atom_types[i];
    const int tj = g_atom_types[j];
    const int pair_block = (ti * meta.num_types + tj) * meta.descriptor_block_size;
    const float x = neighbors.dx[index];
    const float y = neighbors.dy[index];
    const float z = neighbors.dz[index];
    const float r = sqrtf(x * x + y * y + z * z);

    float fn[MAX_NUM_N];
    find_radial_basis(meta.rc_angular, meta.basis_size_angular, r, fn);

    float gn = 0.0f;
    const int c0 = pair_block + meta.num_c_radial + n * (meta.basis_size_angular + 1);
    for (int k = 0; k <= meta.basis_size_angular; ++k) {
      gn += fn[k] * g_c[c0 + k];
    }
    accumulate_s(meta.L_max, r, x, y, z, gn, s);
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const NeighborView neighbors,
  const int* g_atom_types,
  const DescriptorMeta meta,
  const float* __restrict__ g_c,
  float* g_descriptors)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }

  float q[MAX_DIM_ANGULAR] = {0.0f};

  for (int n = 0; n <= meta.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    find_angular_moments(i, n, meta, neighbors, g_atom_types, g_c, s);

    find_q(meta.L_max, meta.L_max, meta.n_max_angular + 1, n, s, q);
  }

  const int radial_dim = meta.n_max_radial + 1;
  for (int n = 0; n <= meta.n_max_angular; ++n) {
    for (int l = 0; l < meta.L_max; ++l) {
      const int ln = l * (meta.n_max_angular + 1) + n;
      g_descriptors[(radial_dim + ln) * N + i] = q[ln];
    }
  }
}

static __global__ void apply_onsite_ann(
  const int N,
  const OnsiteMeta meta,
  const int* __restrict__ g_atom_types,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_ann,
  float* g_onsite)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }

  const int type = g_atom_types[i];
  g_ann += type * meta.block_size;

  const int offset_b0 = meta.hidden * meta.dim;
  const int offset_w1 = offset_b0 + meta.hidden;
  const int offset_b1 = offset_w1 + meta.num_outputs * meta.hidden;

  float out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int mu = 0; mu < meta.num_outputs; ++mu) {
    out[mu] = g_ann[offset_b1 + mu];
  }

  for (int h = 0; h < meta.hidden; ++h) {
    float z = g_ann[offset_b0 + h];
    const int w0 = h * meta.dim;
    for (int d = 0; d < meta.dim; ++d) {
      const float q = g_descriptors[d * N + i] * g_q_scaler[d];
      z += g_ann[w0 + d] * q;
    }
    const float a = tanhf(z);
    for (int mu = 0; mu < meta.num_outputs; ++mu) {
      out[mu] += g_ann[offset_w1 + mu * meta.hidden + h] * a;
    }
  }

  for (int mu = 0; mu < meta.num_outputs; ++mu) {
    g_onsite[meta.num_outputs * i + mu] = out[mu];
  }
}

static __device__ __forceinline__ float hopping_cutoff(const float x)
{
  if (x >= 1.0f) {
    return 0.0f;
  }
  if (x <= 0.0f) {
    return 1.0f;
  }
  const float x2 = x * x;
  const float x3 = x2 * x;
  const float x4 = x3 * x;
  const float x5 = x4 * x;
  return 1.0f - 10.0f * x3 + 15.0f * x4 - 6.0f * x5;
}

static __device__ __forceinline__ int unordered_pair_index_device(int a, int b, const int num_types)
{
  if (a > b) {
    const int t = a;
    a = b;
    b = t;
  }
  return a * num_types - a * (a - 1) / 2 + (b - a);
}

static __device__ __forceinline__ void find_hopping_environment(
  const int i,
  const int j,
  const int num_atoms,
  const HoppingMeta& meta,
  const float* g_descriptors,
  const float* g_q_scaler,
  const float* g_ann,
  float* env)
{
  const int offset_b0 = meta.hidden * meta.dim;
  const int offset_w_env = offset_b0 + meta.hidden;
  const int offset_b_env = offset_w_env + meta.dim_radial * meta.hidden;
  for (int n = 0; n < meta.dim_radial; ++n) {
    env[n] = g_ann[offset_b_env + n];
  }

  for (int h = 0; h < meta.hidden; ++h) {
    float z = g_ann[offset_b0 + h];
    const int w0 = h * meta.dim;
    for (int d = 0; d < meta.dim; ++d) {
      const float qi = g_descriptors[d * num_atoms + i] * g_q_scaler[d];
      const float qj = g_descriptors[d * num_atoms + j] * g_q_scaler[d];
      z += g_ann[w0 + d] * (0.5f * (qi + qj));
    }
    const float a = tanhf(z);
    for (int n = 0; n < meta.dim_radial; ++n) {
      env[n] += g_ann[offset_w_env + n * meta.hidden + h] * a;
    }
  }
}

static __global__ void apply_hopping_ann(
  const int num_bonds,
  const int num_atoms,
  const HoppingMeta meta,
  const int* __restrict__ g_atom_types,
  const int* __restrict__ g_bond_i,
  const int* __restrict__ g_bond_j,
  const float* __restrict__ g_bond_r,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_all_ann,
  const int* __restrict__ g_pair_offsets,
  const float* __restrict__ g_all_c,
  float* g_hopping)
{
  const int b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b >= num_bonds) {
    return;
  }

  const int i = g_bond_i[b];
  const int j = g_bond_j[b];
  const int ti = g_atom_types[i];
  const int tj = g_atom_types[j];
  const int pair_index = unordered_pair_index_device(ti, tj, meta.num_types);
  const int num_outputs = (ti == tj) ? meta.homo_outputs : meta.hetero_outputs;
  const float* g_ann = g_all_ann + g_pair_offsets[pair_index];

  for (int mu = 0; mu < meta.max_outputs; ++mu) {
    g_hopping[meta.max_outputs * b + mu] = 0.0f;
  }

  const int offset_b0 = meta.hidden * meta.dim;
  const int offset_w_env = offset_b0 + meta.hidden;
  const int offset_b_env = offset_w_env + meta.dim_radial * meta.hidden;
  const int offset_w_out = offset_b_env + meta.dim_radial;
  const int offset_b_out = offset_w_out + num_outputs * meta.dim_radial;

  float env[MAX_NUM_N] = {0.0f};
  find_hopping_environment(i, j, num_atoms, meta, g_descriptors, g_q_scaler, g_ann, env);

  const float r = g_bond_r[b];
  float fn[MAX_NUM_N];
  find_radial_basis(meta.rc_radial, meta.basis_size_radial, r, fn);

  float out[21] = {0.0f};
  for (int mu = 0; mu < num_outputs; ++mu) {
    out[mu] = g_ann[offset_b_out + mu];
  }

  const int block_ij = (ti * meta.num_types + tj) * meta.descriptor_block_size;
  const int block_ji = (tj * meta.num_types + ti) * meta.descriptor_block_size;
  for (int n = 0; n < meta.dim_radial; ++n) {
    float gn_ij = 0.0f;
    float gn_ji = 0.0f;
    const int local = n * (meta.basis_size_radial + 1);
    for (int k = 0; k <= meta.basis_size_radial; ++k) {
      gn_ij += fn[k] * g_all_c[block_ij + local + k];
      gn_ji += fn[k] * g_all_c[block_ji + local + k];
    }
    const float gn_pair = 0.5f * (gn_ij + gn_ji);
    const float channel = env[n] * gn_pair;
    for (int mu = 0; mu < num_outputs; ++mu) {
      out[mu] += g_ann[offset_w_out + mu * meta.dim_radial + n] * channel;
    }
  }

  const float fc_hopping = hopping_cutoff(r / meta.rc_hopping);
  for (int mu = 0; mu < num_outputs; ++mu) {
    g_hopping[meta.max_outputs * b + mu] = fc_hopping * out[mu];
  }
}

static __global__ void backward_onsite_ann(
  const int N,
  const OnsiteMeta meta,
  const int* __restrict__ g_atom_types,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_parameters,
  const float* __restrict__ g_grad_onsite,
  float* __restrict__ g_grad_descriptors,
  float* __restrict__ g_grad_parameters)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }

  const int type = g_atom_types[i];
  const float* ann = g_parameters + meta.parameter_offset + type * meta.block_size;
  float* grad_ann = g_grad_parameters + meta.parameter_offset + type * meta.block_size;

  const int offset_b0 = meta.hidden * meta.dim;
  const int offset_w1 = offset_b0 + meta.hidden;
  const int offset_b1 = offset_w1 + meta.num_outputs * meta.hidden;

  for (int h = 0; h < meta.hidden; ++h) {
    float z = ann[offset_b0 + h];
    const int w0 = h * meta.dim;
    for (int d = 0; d < meta.dim; ++d) {
      const float q = g_descriptors[d * N + i] * g_q_scaler[d];
      z += ann[w0 + d] * q;
    }
    const float a = tanhf(z);

    float grad_a = 0.0f;
    for (int mu = 0; mu < meta.num_outputs; ++mu) {
      const float gy = g_grad_onsite[meta.num_outputs * i + mu];
      atomicAdd(&grad_ann[offset_w1 + mu * meta.hidden + h], gy * a);
      grad_a += gy * ann[offset_w1 + mu * meta.hidden + h];
      if (h == 0) {
        atomicAdd(&grad_ann[offset_b1 + mu], gy);
      }
    }

    const float grad_z = grad_a * (1.0f - a * a);
    atomicAdd(&grad_ann[offset_b0 + h], grad_z);
    for (int d = 0; d < meta.dim; ++d) {
      const float q = g_descriptors[d * N + i] * g_q_scaler[d];
      atomicAdd(&grad_ann[w0 + d], grad_z * q);
      atomicAdd(&g_grad_descriptors[d * N + i], grad_z * ann[w0 + d] * g_q_scaler[d]);
    }
  }
}

static __global__ void backward_hopping_ann(
  const int num_bonds,
  const int num_atoms,
  const HoppingMeta meta,
  const int* __restrict__ g_atom_types,
  const int* __restrict__ g_bond_i,
  const int* __restrict__ g_bond_j,
  const float* __restrict__ g_bond_r,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_parameters,
  const int* __restrict__ g_pair_offsets,
  const float* __restrict__ g_grad_hopping,
  float* __restrict__ g_grad_descriptors,
  float* __restrict__ g_grad_parameters)
{
  const int b = threadIdx.x + blockIdx.x * blockDim.x;
  if (b >= num_bonds) {
    return;
  }

  const int i = g_bond_i[b];
  const int j = g_bond_j[b];
  const int ti = g_atom_types[i];
  const int tj = g_atom_types[j];
  const int pair_index = unordered_pair_index_device(ti, tj, meta.num_types);
  const int num_outputs = (ti == tj) ? meta.homo_outputs : meta.hetero_outputs;

  const float* ann = g_parameters + meta.parameter_offset + g_pair_offsets[pair_index];
  float* grad_ann = g_grad_parameters + meta.parameter_offset + g_pair_offsets[pair_index];
  const float* all_c = g_parameters + meta.descriptor_offset;
  float* grad_c = g_grad_parameters + meta.descriptor_offset;

  const int offset_b0 = meta.hidden * meta.dim;
  const int offset_w_env = offset_b0 + meta.hidden;
  const int offset_b_env = offset_w_env + meta.dim_radial * meta.hidden;
  const int offset_w_out = offset_b_env + meta.dim_radial;
  const int offset_b_out = offset_w_out + num_outputs * meta.dim_radial;

  const float r = g_bond_r[b];
  const float fc_hopping = hopping_cutoff(r / meta.rc_hopping);

  float env[MAX_NUM_N] = {0.0f};
  float grad_env[MAX_NUM_N] = {0.0f};
  float gn_pair[MAX_NUM_N] = {0.0f};
  float grad_gn_pair[MAX_NUM_N] = {0.0f};
  find_hopping_environment(i, j, num_atoms, meta, g_descriptors, g_q_scaler, ann, env);

  float fn[MAX_NUM_N];
  find_radial_basis(meta.rc_radial, meta.basis_size_radial, r, fn);
  const int block_ij = (ti * meta.num_types + tj) * meta.descriptor_block_size;
  const int block_ji = (tj * meta.num_types + ti) * meta.descriptor_block_size;
  for (int n = 0; n < meta.dim_radial; ++n) {
    float gij = 0.0f;
    float gji = 0.0f;
    const int local = n * (meta.basis_size_radial + 1);
    for (int k = 0; k <= meta.basis_size_radial; ++k) {
      gij += fn[k] * all_c[block_ij + local + k];
      gji += fn[k] * all_c[block_ji + local + k];
    }
    gn_pair[n] = 0.5f * (gij + gji);
  }

  for (int mu = 0; mu < num_outputs; ++mu) {
    const float gy = g_grad_hopping[meta.max_outputs * b + mu] * fc_hopping;
    atomicAdd(&grad_ann[offset_b_out + mu], gy);
    for (int n = 0; n < meta.dim_radial; ++n) {
      const float channel = env[n] * gn_pair[n];
      atomicAdd(&grad_ann[offset_w_out + mu * meta.dim_radial + n], gy * channel);
      const float gc = gy * ann[offset_w_out + mu * meta.dim_radial + n];
      grad_env[n] += gc * gn_pair[n];
      grad_gn_pair[n] += gc * env[n];
    }
  }

  for (int n = 0; n < meta.dim_radial; ++n) {
    const int local = n * (meta.basis_size_radial + 1);
    for (int k = 0; k <= meta.basis_size_radial; ++k) {
      const float contribution = 0.5f * grad_gn_pair[n] * fn[k];
      atomicAdd(&grad_c[block_ij + local + k], contribution);
      atomicAdd(&grad_c[block_ji + local + k], contribution);
    }
    atomicAdd(&grad_ann[offset_b_env + n], grad_env[n]);
  }

  for (int h = 0; h < meta.hidden; ++h) {
    float z = ann[offset_b0 + h];
    const int w0 = h * meta.dim;
    for (int d = 0; d < meta.dim; ++d) {
      const float qi = g_descriptors[d * num_atoms + i] * g_q_scaler[d];
      const float qj = g_descriptors[d * num_atoms + j] * g_q_scaler[d];
      z += ann[w0 + d] * (0.5f * (qi + qj));
    }
    const float a = tanhf(z);
    float grad_a = 0.0f;
    for (int n = 0; n < meta.dim_radial; ++n) {
      atomicAdd(&grad_ann[offset_w_env + n * meta.hidden + h], grad_env[n] * a);
      grad_a += grad_env[n] * ann[offset_w_env + n * meta.hidden + h];
    }
    const float grad_z = grad_a * (1.0f - a * a);
    atomicAdd(&grad_ann[offset_b0 + h], grad_z);
    for (int d = 0; d < meta.dim; ++d) {
      const float qi = g_descriptors[d * num_atoms + i] * g_q_scaler[d];
      const float qj = g_descriptors[d * num_atoms + j] * g_q_scaler[d];
      const float qpair = 0.5f * (qi + qj);
      atomicAdd(&grad_ann[w0 + d], grad_z * qpair);
      const float gq = 0.5f * grad_z * ann[w0 + d] * g_q_scaler[d];
      atomicAdd(&g_grad_descriptors[d * num_atoms + i], gq);
      atomicAdd(&g_grad_descriptors[d * num_atoms + j], gq);
    }
  }
}

static __global__ void backward_descriptor_radial(
  const int N,
  const DescriptorMeta meta,
  const NeighborView neighbors,
  const int* __restrict__ g_atom_types,
  const float* __restrict__ g_grad_descriptors,
  float* __restrict__ grad_c)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }
  const int ti = g_atom_types[i];
  for (int m = 0; m < neighbors.nn[i]; ++m) {
    const int index = neighbors.nn_sum[i] + m;
    const int j = neighbors.nl[index];
    const int tj = g_atom_types[j];
    const int pair_block = (ti * meta.num_types + tj) * meta.descriptor_block_size;
    const float x = neighbors.dx[index], y = neighbors.dy[index], z = neighbors.dz[index];
    const float r = sqrtf(x * x + y * y + z * z);
    float fn[MAX_NUM_N];
    find_radial_basis(meta.rc_radial, meta.basis_size_radial, r, fn);
    for (int n = 0; n <= meta.n_max_radial; ++n) {
      const float gq = g_grad_descriptors[n * N + i];
      const int c0 = pair_block + n * (meta.basis_size_radial + 1);
      for (int k = 0; k <= meta.basis_size_radial; ++k) {
        atomicAdd(&grad_c[c0 + k], gq * fn[k]);
      }
    }
  }
}

static __global__ void backward_descriptor_angular(
  const int N,
  const DescriptorMeta meta,
  const NeighborView neighbors,
  const int* __restrict__ g_atom_types,
  const float* __restrict__ all_c,
  const float* __restrict__ g_grad_descriptors,
  float* __restrict__ grad_c)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }
  const int ti = g_atom_types[i];

  for (int n = 0; n <= meta.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    find_angular_moments(i, n, meta, neighbors, g_atom_types, all_c, s);

    float omega[NUM_OF_ABC] = {0.0f};
    for (int ell = 0; ell < meta.L_max; ++ell) {
      const int L = ell + 1;
      const float beta =
        g_grad_descriptors[((meta.n_max_radial + 1) + ell * (meta.n_max_angular + 1) + n) * N + i];
      const int start = L * L - 1;
      omega[start] += beta * 2.0f * C3B[start] * s[start];
      for (int a = 1; a <= 2 * L; ++a) {
        omega[start + a] += beta * 4.0f * C3B[start + a] * s[start + a];
      }
    }

    for (int m = 0; m < neighbors.nn[i]; ++m) {
      const int index = neighbors.nn_sum[i] + m;
      const int j = neighbors.nl[index];
      const int tj = g_atom_types[j];
      const int pair_block = (ti * meta.num_types + tj) * meta.descriptor_block_size;
      const float x = neighbors.dx[index], y = neighbors.dy[index], z = neighbors.dz[index];
      const float r = sqrtf(x * x + y * y + z * z);
      float fn[MAX_NUM_N];
      find_radial_basis(meta.rc_angular, meta.basis_size_angular, r, fn);
      const int c0 = pair_block + meta.num_c_radial + n * (meta.basis_size_angular + 1);
      for (int k = 0; k <= meta.basis_size_angular; ++k) {
        float moment[NUM_OF_ABC] = {0.0f};
        accumulate_s(meta.L_max, r, x, y, z, fn[k], moment);
        float contribution = 0.0f;
        const int num_abc = (meta.L_max + 1) * (meta.L_max + 1) - 1;
        for (int abc = 0; abc < num_abc; ++abc) {
          contribution += omega[abc] * moment[abc];
        }
        atomicAdd(&grad_c[c0 + k], contribution);
      }
    }
  }
}

template <typename T>
void copy_to_gpu(const std::vector<T>& source, GPU_Vector<T>& target)
{
  target.resize(source.size());
  target.copy_from_host(source.data());
}

void copy_to_gpu(const std::vector<double>& source, GPU_Vector<float>& target)
{
  const std::vector<float> buffer(source.begin(), source.end());
  copy_to_gpu(buffer, target);
}

} // namespace

void TB::NeighborList::initialize(const int num_atoms, const std::vector<Neighbor>& neighbors)
{
  std::vector<int> counts(num_atoms, 0), offsets(num_atoms);
  for (const auto& neighbor : neighbors) {
    if (neighbor.i < 0 || neighbor.i >= num_atoms || neighbor.j < 0 || neighbor.j >= num_atoms) {
      PRINT_INPUT_ERROR("Invalid atom index in neighbor list.");
    }
    ++counts[neighbor.i];
  }

  int total = 0;
  for (int i = 0; i < num_atoms; ++i) {
    offsets[i] = total;
    total += counts[i];
  }

  std::vector<int> indices(total), cursor = offsets;
  std::vector<float> x(total), y(total), z(total);
  for (const auto& neighbor : neighbors) {
    const int p = cursor[neighbor.i]++;
    indices[p] = neighbor.j;
    x[p] = static_cast<float>(neighbor.dx);
    y[p] = static_cast<float>(neighbor.dy);
    z[p] = static_cast<float>(neighbor.dz);
  }
  copy_to_gpu(offsets, nn_sum_);
  copy_to_gpu(counts, nn_);
  copy_to_gpu(indices, nl_);
  copy_to_gpu(x, dx_);
  copy_to_gpu(y, dy_);
  copy_to_gpu(z, dz_);
}

void TB::copy_neighbor_lists_to_gpu()
{
  radial_.initialize(num_atoms_, dataset_.neighbors_radial);
  angular_.initialize(num_atoms_, dataset_.neighbors_angular);

  std::vector<int> bi(num_bonds_);
  std::vector<int> bj(num_bonds_);
  std::vector<float> br(num_bonds_);
  for (int b = 0; b < num_bonds_; ++b) {
    bi[b] = dataset_.hopping_bonds[b].i;
    bj[b] = dataset_.hopping_bonds[b].j;
    br[b] = static_cast<float>(dataset_.hopping_bonds[b].r);
  }
  copy_to_gpu(bi, bond_i_);
  copy_to_gpu(bj, bond_j_);
  copy_to_gpu(br, bond_r_);
}

void TB::compute_descriptors(const float* descriptor_parameters)
{
  const DescriptorMeta meta = descriptor_meta(para_);

  const int block_size = 128;
  const int grid_size = (num_atoms_ + block_size - 1) / block_size;

  find_descriptors_radial<<<grid_size, block_size>>>(
    num_atoms_,
    NeighborView{
      radial_.nn_sum_.data(),
      radial_.nn_.data(),
      radial_.nl_.data(),
      radial_.dx_.data(),
      radial_.dy_.data(),
      radial_.dz_.data()},
    atom_types_.data(),
    meta,
    descriptor_parameters,
    descriptors_.data());
  GPU_CHECK_KERNEL

  find_descriptors_angular<<<grid_size, block_size>>>(
    num_atoms_,
    NeighborView{
      angular_.nn_sum_.data(),
      angular_.nn_.data(),
      angular_.nl_.data(),
      angular_.dx_.data(),
      angular_.dy_.data(),
      angular_.dz_.data()},
    atom_types_.data(),
    meta,
    descriptor_parameters,
    descriptors_.data());
  GPU_CHECK_KERNEL
}

void TB::initialize_q_scaler()
{
  GPU_Vector<float> reference_c(para_.number_of_variables_descriptor, 1.0f);
  compute_descriptors(reference_c.data());

  std::vector<float> q(static_cast<std::size_t>(num_atoms_) * para_.dim);
  descriptors_.copy_to_host(q.data());
  reference_descriptor_max_abs_.assign(para_.dim, 0.0f);
  std::vector<float> scaler(para_.dim, 1.0f);
  for (int d = 0; d < para_.dim; ++d) {
    double max_abs = 0.0;
    for (int i = 0; i < num_atoms_; ++i) {
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(q[d * num_atoms_ + i])));
    }
    reference_descriptor_max_abs_[d] = static_cast<float>(max_abs);
    if (max_abs > 1.0e-12) {
      scaler[d] = static_cast<float>(1.0 / max_abs);
    }
  }
  q_scaler_.copy_from_host(scaler.data());
}

void TB::compute_model(const double* parameters)
{
  for (int v = 0; v < para_.number_of_variables; ++v) {
    if (
      !std::isfinite(parameters[v]) ||
      std::fabs(parameters[v]) > std::numeric_limits<float>::max()) {
      PRINT_INPUT_ERROR("TB parameter is outside the finite single-precision GPU range.");
    }
    parameters_host_[v] = static_cast<float>(parameters[v]);
  }
  parameters_.copy_from_host(parameters_host_.data());

  const float* g_descriptor = parameters_.data() + para_.offset_descriptor;
  compute_descriptors(g_descriptor);

  const int block_size = 128;
  const int atom_grid = (num_atoms_ + block_size - 1) / block_size;
  apply_onsite_ann<<<atom_grid, block_size>>>(
    num_atoms_,
    onsite_meta(para_),
    atom_types_.data(),
    descriptors_.data(),
    q_scaler_.data(),
    parameters_.data() + para_.offset_onsite,
    onsite_.data());
  GPU_CHECK_KERNEL

  const int bond_grid = (num_bonds_ + block_size - 1) / block_size;
  apply_hopping_ann<<<bond_grid, block_size>>>(
    num_bonds_,
    num_atoms_,
    hopping_meta(para_),
    atom_types_.data(),
    bond_i_.data(),
    bond_j_.data(),
    bond_r_.data(),
    descriptors_.data(),
    q_scaler_.data(),
    parameters_.data() + para_.offset_hopping,
    hopping_pair_offsets_.data(),
    g_descriptor,
    hopping_.data());
  GPU_CHECK_KERNEL

  onsite_.copy_to_host(onsite_host_.data());
  hopping_.copy_to_host(hopping_host_.data());
}

void TB::get_q_scaler(std::vector<float>& scaler)
{
  scaler.resize(para_.dim);
  q_scaler_.copy_to_host(scaler.data());
}

void TB::get_reference_descriptor_max_abs(std::vector<float>& max_abs) const
{
  max_abs = reference_descriptor_max_abs_;
}

void TB::set_q_scaler(const std::vector<float>& scaler)
{
  if (static_cast<int>(scaler.size()) != para_.dim) {
    PRINT_INPUT_ERROR("q_scaler has the wrong descriptor dimension.");
  }
  q_scaler_.copy_from_host(scaler.data());
}

void TB::initialize_parameters(const Parameters& para, std::vector<double>& parameters)
{
  std::mt19937 rng(static_cast<unsigned int>(para.seed));
  parameters.assign(para.number_of_variables, 0.0f);
  const int D = para.dim;
  const int Dr = para.dim_radial;

  {
    const int H = para.num_neurons_onsite;
    const float s0 = 1.0f / std::sqrt(static_cast<float>(D + H));
    const float s1 = 1.0f / std::sqrt(static_cast<float>(H + 1));
    std::normal_distribution<float> w0(0.0f, s0), w1(0.0f, s1);
    for (int t = 0; t < para.num_types; ++t) {
      const int base = para.offset_onsite + t * para.onsite_block_size;
      const int b0 = base + H * D;
      const int w1off = b0 + H;
      const int b1 = w1off + para.num_onsite_outputs * H;
      for (int i = base; i < b0; ++i) {
        parameters[i] = w0(rng);
      }
      for (int i = w1off; i < b1; ++i) {
        parameters[i] = w1(rng);
      }
    }
  }

  {
    const int H = para.num_neurons_hopping;
    const float s0 = 1.0f / std::sqrt(static_cast<float>(D + H));
    const float se = 1.0f / std::sqrt(static_cast<float>(H + Dr));
    std::normal_distribution<float> w0(0.0f, s0), we(0.0f, se);
    for (int pair = 0; pair < para.num_element_pairs; ++pair) {
      const int outputs = para.hopping_pair_outputs[pair];
      const int base = para.offset_hopping + para.hopping_pair_offsets[pair];
      const int b0 = base + H * D;
      const int wenv = b0 + H;
      const int benv = wenv + Dr * H;
      const int wout = benv + Dr;
      const int bout = wout + outputs * Dr;
      std::normal_distribution<float> wo(0.0f, 1.0f / std::sqrt(static_cast<float>(Dr + outputs)));
      for (int i = base; i < b0; ++i) {
        parameters[i] = w0(rng);
      }
      for (int i = wenv; i < benv; ++i) {
        parameters[i] = we(rng);
      }
      for (int i = wout; i < bout; ++i) {
        parameters[i] = wo(rng);
      }
    }
  }

  std::uniform_real_distribution<float> descriptor_dist(-1.0f, 1.0f);
  for (int v = para.offset_descriptor; v < para.number_of_variables; ++v) {
    parameters[v] = descriptor_dist(rng);
  }
}

void TB::write_model_header(std::ostream& output) const
{
  output << std::setprecision(17);
  output << "tb 2\n";
  output << "type " << para_.num_types;
  for (const auto& element : para_.elements) {
    output << " " << element;
  }
  output << "\n";
  output << "basis " << para_.basis << "\n";
  output << "orbitals_per_atom " << para_.orbitals_per_atom << "\n";
  if (para_.is_sp3()) {
    output << "orbital_order s px py pz\n";
    output << "onsite_outputs 2 epsilon_s epsilon_p\n";
    output << "homo_hopping_outputs 4 ss_sigma sp_sigma pp_sigma pp_pi\n";
    output << "hetero_hopping_outputs 5 ss_sigma sp_sigma ps_sigma pp_sigma pp_pi\n";
  } else if (para_.is_sp3sstar()) {
    output << "orbital_order s px py pz sstar\n";
    output << "onsite_outputs 3 epsilon_s epsilon_p epsilon_sstar\n";
    output << "homo_hopping_outputs 7 ss_sigma sp_sigma pp_sigma pp_pi ssstar_sigma sstarp_sigma "
              "sstarsstar_sigma\n";
    output << "hetero_hopping_outputs 10 ss_sigma sp_sigma ps_sigma pp_sigma pp_pi ssstar_sigma "
              "sstars_sigma sstarp_sigma psstar_sigma sstarsstar_sigma\n";
  } else if (para_.is_sp3d5()) {
    output << "orbital_order s px py pz dxy dyz dz2 dxz dx2_y2\n";
    output << "onsite_outputs 3 epsilon_s epsilon_p epsilon_d\n";
    output << "homo_hopping_outputs 10 ss_sigma sp_sigma pp_sigma pp_pi sd_sigma pd_sigma pd_pi "
              "dd_sigma dd_pi dd_delta\n";
    output << "hetero_hopping_outputs 14 ss_sigma sp_sigma ps_sigma pp_sigma pp_pi sd_sigma "
              "ds_sigma pd_sigma pd_pi dp_sigma dp_pi dd_sigma dd_pi dd_delta\n";
  } else {
    output << "orbital_order s px py pz dxy dyz dz2 dxz dx2_y2 sstar\n";
    output << "onsite_outputs 4 epsilon_s epsilon_p epsilon_d epsilon_sstar\n";
    output << "homo_hopping_outputs 14 ss_sigma sp_sigma pp_sigma pp_pi sd_sigma pd_sigma pd_pi "
              "dd_sigma dd_pi dd_delta ssstar_sigma sstarp_sigma sstard_sigma sstarsstar_sigma\n";
    output << "hetero_hopping_outputs 21 ss_sigma sp_sigma ps_sigma pp_sigma pp_pi sd_sigma "
              "ds_sigma pd_sigma pd_pi dp_sigma dp_pi dd_sigma dd_pi dd_delta ssstar_sigma "
              "sstars_sigma sstarp_sigma psstar_sigma sstard_sigma dsstar_sigma sstarsstar_sigma\n";
  }
  output << "hetero_channel_order canonical_low_type_to_high_type\n";
  output << "cutoff " << para_.rc_radial << " " << para_.rc_angular << " " << para_.rc_hopping
         << "\n";
  output << "n_max " << para_.n_max_radial << " " << para_.n_max_angular << "\n";
  output << "basis_size " << para_.basis_size_radial << " " << para_.basis_size_angular << "\n";
  output << "l_max " << para_.L_max << "\n";
  output << "neuron " << para_.num_neurons_onsite << " " << para_.num_neurons_hopping << "\n";
  output << "descriptor_parameterization ordered_center_neighbor_pairs\n";
  output << "hopping_pair_parameterization unordered_element_pairs\n";
  output << "hopping_factorization env_ann_times_symmetric_pair_gn\n";
  output << "band_range " << para_.band_min << " " << para_.band_max << "\n";
  for (int n = 0; n < (para_.band_max - para_.band_min); ++n) {
    output << "band_weight " << (para_.band_min + n) << " " << (para_.band_min + n + 1) << " "
           << para_.band_weights[n] << "\n";
  }
  output << "dispersion " << para_.dispersion_weight << " ";
  if (std::isinf(para_.dispersion_sigma)) {
    output << "inf\n";
  } else {
    output << para_.dispersion_sigma << "\n";
  }
}

void TB::write_model_file(const std::string& filename, const double* parameters)
{
  std::ofstream output(filename);
  if (!output.is_open()) {
    PRINT_INPUT_ERROR(("Failed to open " + filename + " for writing.").c_str());
  }

  write_model_header(output);
  output << "parameters " << para_.number_of_variables << "\n";
  for (int v = 0; v < para_.number_of_variables; ++v) {
    output << std::scientific << std::setprecision(16) << parameters[v] << "\n";
  }

  std::vector<float> scaler;
  get_q_scaler(scaler);
  output << "q_scaler " << scaler.size() << "\n";
  for (float value : scaler) {
    output << std::scientific << std::setprecision(9) << value << "\n";
  }
}

void TB::read_model_file(
  const std::string& filename, std::vector<double>& parameters, std::vector<float>& scaler) const
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR(("Failed to open " + filename + ".").c_str());
  }
  auto next = [&]() {
    std::string token;
    if (!(input >> token)) {
      PRINT_INPUT_ERROR(("Incomplete model in " + filename + ".").c_str());
    }
    return token;
  };
  auto expect = [&](const std::string& expected, bool real_header_value) {
    const std::string actual = next();
    if (actual == expected) {
      return;
    }
    char* end_actual = nullptr;
    char* end_expected = nullptr;
    errno = 0;
    const double a = std::strtod(actual.c_str(), &end_actual);
    const bool actual_in_range = errno != ERANGE;
    const double b = std::strtod(expected.c_str(), &end_expected);
    if (
      end_actual != actual.c_str() && *end_actual == '\0' && end_expected != expected.c_str() &&
      *end_expected == '\0' && actual_in_range && std::isfinite(a) && std::isfinite(b)) {
      if (a == b) {
        return;
      }
      if (real_header_value && std::fabs(b) <= std::numeric_limits<float>::max()) {
        std::ostringstream previous;
        previous << std::setprecision(10) << static_cast<float>(b);
        if (actual == previous.str()) {
          return;
        }
      }
    }
    PRINT_INPUT_ERROR((filename + ": expected '" + expected + "', got '" + actual +
                       "'; the model must match tb.in.")
                        .c_str());
  };
  std::stringstream header;
  write_model_header(header);
  std::string line;
  while (std::getline(header, line)) {
    const auto tokens = get_tokens(line);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const bool real_value = i > 0 && (tokens[0] == "cutoff" || tokens[0] == "dispersion" ||
                                        (tokens[0] == "band_weight" && i == 3));
      expect(tokens[i], real_value);
    }
  }
  auto read_values = [&](const char* label, int count) {
    expect(label, false);
    expect(std::to_string(count), false);
    std::vector<double> values(count);
    for (double& value : values) {
      const auto token = next();
      char* end = nullptr;
      errno = 0;
      value = std::strtod(token.c_str(), &end);
      if (
        end == token.c_str() || *end != '\0' || !std::isfinite(value) ||
        std::fabs(value) > std::numeric_limits<float>::max() || (errno == ERANGE && value == 0.0)) {
        PRINT_INPUT_ERROR((filename + ": invalid " + label + " value '" + token + "'.").c_str());
      }
    }
    return values;
  };
  parameters = read_values("parameters", para_.number_of_variables);
  const auto saved_scaler = read_values("q_scaler", para_.dim);
  scaler.assign(saved_scaler.begin(), saved_scaler.end());
  for (float value : scaler) {
    if (value <= 0.0f) {
      PRINT_INPUT_ERROR((filename + ": q_scaler values must be positive.").c_str());
    }
  }

  std::string extra;
  if (input >> extra || input.bad()) {
    PRINT_INPUT_ERROR((filename + ": unexpected data after q_scaler.").c_str());
  }
}

namespace
{

void check_solver(const int status, const char* operation)
{
  if (status != 0) {
    PRINT_INPUT_ERROR((std::string("TB eigensolver failed to ") + operation + " (status " +
                       std::to_string(status) + ").")
                        .c_str());
  }
}

inline std::size_t matrix_index(const int k, const int row, const int col, const int n)
{
  return static_cast<std::size_t>(k) * n * n + static_cast<std::size_t>(col) * n + row;
}

void check_direction(const double l, const double m, const double n)
{
  const double norm2 = l * l + m * m + n * n;
  if (!(norm2 > 0.0)) {
    PRINT_INPUT_ERROR("Slater-Koster direction has zero norm.");
  }
  if (std::fabs(norm2 - 1.0) > 1.0e-5) {
    PRINT_INPUT_ERROR("Slater-Koster direction cosines are not normalized.");
  }
}

std::array<double, 5> make_d_sigma_vector(const double l, const double m, const double n)
{
  constexpr double SQRT3 = 1.7320508075688772935;
  return {
    {SQRT3 * l * m,
     SQRT3 * m * n,
     0.5 * (3.0 * n * n - 1.0),
     SQRT3 * l * n,
     0.5 * SQRT3 * (l * l - m * m)}};
}

std::array<double, 15>
make_pd_pi_matrix(const double l, const double m, const double n, const std::array<double, 5>& q)
{
  constexpr double SQRT3 = 1.7320508075688772935;
  const double u[3] = {l, m, n};

  const double Du[5][3] = {
    {0.5 * SQRT3 * m, 0.5 * SQRT3 * l, 0.0},
    {0.0, 0.5 * SQRT3 * n, 0.5 * SQRT3 * m},
    {-0.5 * l, -0.5 * m, n},
    {0.5 * SQRT3 * n, 0.0, 0.5 * SQRT3 * l},
    {0.5 * SQRT3 * l, -0.5 * SQRT3 * m, 0.0}};

  std::array<double, 15> b{};
  for (int alpha = 0; alpha < 3; ++alpha) {
    for (int A = 0; A < 5; ++A) {
      b[5 * alpha + A] = (2.0 / SQRT3) * (Du[A][alpha] - q[A] * u[alpha]);
    }
  }
  return b;
}

void make_block(
  const int orbitals_per_atom,
  const int type_i,
  const int type_j,
  const double l,
  const double m,
  const double n,
  const float* hopping,
  double* block)
{
  if (
    orbitals_per_atom != 4 && orbitals_per_atom != 5 && orbitals_per_atom != 9 &&
    orbitals_per_atom != 10) {
    PRINT_INPUT_ERROR("Unsupported orbital count in SK block construction.");
  }
  check_direction(l, m, n);
  const bool has_d = orbitals_per_atom >= 9;
  const bool has_sstar = orbitals_per_atom == 5 || orbitals_per_atom == 10;
  const int outputs = has_d ? (has_sstar ? 21 : 14) : (has_sstar ? 10 : 5);
  const int sstar_offset = has_d ? 14 : 5;
  const int homo_index[2][21] = {
    {0, 1, 1, 2, 3, 4, 4, 5, 5, 6},
    {0, 1, 1, 2, 3, 4, 4, 5, 6, 5, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 13}};
  double h[21];
  for (int i = 0; i < outputs; ++i) {
    h[i] = hopping[type_i == type_j ? homo_index[has_d][i] : i];
  }
  if (type_i > type_j) {
    std::swap(h[1], h[2]);
    if (has_d) {
      std::swap(h[5], h[6]);
      std::swap(h[7], h[9]);
      std::swap(h[8], h[10]);
    }
    if (has_sstar) {
      std::swap(h[sstar_offset], h[sstar_offset + 1]);
      std::swap(h[sstar_offset + 2], h[sstar_offset + 3]);
      if (has_d) {
        std::swap(h[sstar_offset + 4], h[sstar_offset + 5]);
      }
    }
  }

  std::fill_n(block, orbitals_per_atom * orbitals_per_atom, 0.0);
  auto at = [block, orbitals_per_atom](const int row, const int col) -> double& {
    return block[orbitals_per_atom * row + col];
  };
  const double u[3] = {l, m, n};
  at(0, 0) = h[0];
  for (int a = 0; a < 3; ++a) {
    at(0, a + 1) = -u[a] * h[1];
    at(a + 1, 0) = +u[a] * h[2];
  }
  const double delta_pp = h[3] - h[4];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      at(a + 1, b + 1) = (a == b ? h[4] : 0.0) + u[a] * u[b] * delta_pp;
    }
  }

  std::array<double, 5> q{};
  if (has_d) {
    q = make_d_sigma_vector(l, m, n);
    for (int A = 0; A < 5; ++A) {
      at(0, A + 4) = h[5] * q[A];
      at(A + 4, 0) = h[6] * q[A];
    }
    const auto bpi = make_pd_pi_matrix(l, m, n, q);
    for (int a = 0; a < 3; ++a) {
      for (int A = 0; A < 5; ++A) {
        const double bsigma = u[a] * q[A];
        at(a + 1, A + 4) = -(h[7] * bsigma + h[8] * bpi[5 * a + A]);
        at(A + 4, a + 1) = +(h[9] * bsigma + h[10] * bpi[5 * a + A]);
      }
    }
    for (int A = 0; A < 5; ++A) {
      for (int B = 0; B < 5; ++B) {
        const double sigma = q[A] * q[B];
        double pi = 0.0;
        for (int alpha = 0; alpha < 3; ++alpha) {
          pi += bpi[5 * alpha + A] * bpi[5 * alpha + B];
        }
        const double delta = (A == B ? 1.0 : 0.0) - sigma - pi;
        at(A + 4, B + 4) = h[11] * sigma + h[12] * pi + h[13] * delta;
      }
    }
  }

  if (has_sstar) {
    const int sstar = orbitals_per_atom - 1;
    at(0, sstar) = h[sstar_offset];
    at(sstar, 0) = h[sstar_offset + 1];
    for (int a = 0; a < 3; ++a) {
      at(sstar, a + 1) = -u[a] * h[sstar_offset + 2];
      at(a + 1, sstar) = +u[a] * h[sstar_offset + 3];
    }
    if (has_d) {
      for (int A = 0; A < 5; ++A) {
        at(sstar, A + 4) = h[sstar_offset + 4] * q[A];
        at(A + 4, sstar) = h[sstar_offset + 5] * q[A];
      }
    }
    at(sstar, sstar) = h[outputs - 1];
  }
}

} // namespace

struct TB::Eigensolver {
  std::size_t N_;
  std::size_t batch_size_;
  std::size_t M_;

  GPU_Vector<tbComplex> A_;
  std::vector<tbComplex> A_cpu_;
  GPU_Vector<float> W_;

  gpusolverDnHandle_t handle_ = nullptr;
  gpusolverSyevjInfo_t para_ = nullptr;
  int lwork_[2] = {0, 0};
  GPU_Vector<tbComplex> work_;
  GPU_Vector<int> info_;
  std::vector<int> info_cpu_;

  Eigensolver(std::size_t N_in, std::size_t batch_size_in)
    : N_(N_in),
      batch_size_(batch_size_in),
      M_(N_ * N_ * batch_size_),
      A_(M_),
      A_cpu_(M_),
      W_(N_ * batch_size_),
      info_(batch_size_),
      info_cpu_(batch_size_)
  {
    if (N_ == 0 || batch_size_ == 0) {
      PRINT_INPUT_ERROR("TB eigensolver requires nonzero dimensions.");
    }

    check_solver(gpusolverDnCreate(&handle_), "create handle");
    check_solver(gpusolverDnCreateSyevjInfo(&para_), "create Jacobi parameters");

    for (int vectors = 0; vectors < 2; ++vectors) {
      check_solver(
        tbsolverDnCheevjBatched_bufferSize(
          handle_,
          vectors ? GPUSOLVER_EIG_MODE_VECTOR : GPUSOLVER_EIG_MODE_NOVECTOR,
          GPUSOLVER_FILL_MODE_LOWER,
          static_cast<int>(N_),
          A_.data(),
          static_cast<int>(N_),
          W_.data(),
          &lwork_[vectors],
          para_,
          static_cast<int>(batch_size_)),
        "query workspace");
    }
    work_.resize(std::max(lwork_[0], lwork_[1]));
  }

  ~Eigensolver()
  {
    if (para_ != nullptr) {
      check_solver(gpusolverDnDestroySyevjInfo(para_), "destroy Jacobi parameters");
    }
    if (handle_ != nullptr) {
      check_solver(gpusolverDnDestroy(handle_), "destroy handle");
    }
  }

  void solve(
    bool vectors,
    const double* AR,
    const double* AI,
    float* W_cpu,
    float* VR_cpu = nullptr,
    float* VI_cpu = nullptr)
  {
    for (std::size_t n = 0; n < M_; ++n) {
      A_cpu_[n].x = static_cast<float>(AR[n]);
      A_cpu_[n].y = static_cast<float>(AI[n]);
    }
    A_.copy_from_host(A_cpu_.data());

    check_solver(
      tbsolverDnCheevjBatched(
        handle_,
        vectors ? GPUSOLVER_EIG_MODE_VECTOR : GPUSOLVER_EIG_MODE_NOVECTOR,
        GPUSOLVER_FILL_MODE_LOWER,
        static_cast<int>(N_),
        A_.data(),
        static_cast<int>(N_),
        W_.data(),
        work_.data(),
        lwork_[vectors],
        info_.data(),
        para_,
        static_cast<int>(batch_size_)),
      "solve eigenproblem");

    W_.copy_to_host(W_cpu);
    if (vectors) {
      A_.copy_to_host(A_cpu_.data());
    }
    info_.copy_to_host(info_cpu_.data());
    for (std::size_t b = 0; b < batch_size_; ++b) {
      if (info_cpu_[b] != 0) {
        PRINT_INPUT_ERROR(
          vectors ? "TB eigensolver failed to converge in eigenvector mode."
                  : "TB eigensolver failed to converge in eigenvalue-only mode.");
      }
    }
    if (vectors) {
      for (std::size_t n = 0; n < M_; ++n) {
        VR_cpu[n] = A_cpu_[n].x;
        VI_cpu[n] = A_cpu_[n].y;
      }
    }
  }
};

TB::TB(const Parameters& para, const Dataset& dataset) : para_(para), dataset_(dataset)
{
  num_atoms_ = dataset_.structure.num_atoms;
  num_bonds_ = static_cast<int>(dataset_.hopping_bonds.size());
  num_kpoints_ = static_cast<int>(dataset_.kpoints.size());
  orbitals_per_atom_ = para_.orbitals_per_atom;
  num_orbitals_ = orbitals_per_atom_ * num_atoms_;

  if (num_atoms_ <= 0) {
    PRINT_INPUT_ERROR("TB requires at least one atom.");
  }
  if (num_kpoints_ <= 0) {
    PRINT_INPUT_ERROR("TB requires at least one k-point.");
  }
  if (
    orbitals_per_atom_ != 4 && orbitals_per_atom_ != 5 && orbitals_per_atom_ != 9 &&
    orbitals_per_atom_ != 10) {
    PRINT_INPUT_ERROR("Unsupported number of orbitals per atom.");
  }

  std::vector<int> atom_types(num_atoms_);
  for (int i = 0; i < num_atoms_; ++i) {
    atom_types[i] = dataset_.structure.atoms[i].type;
  }
  copy_to_gpu(atom_types, atom_types_);

  copy_neighbor_lists_to_gpu();

  descriptors_.resize(static_cast<std::size_t>(num_atoms_) * para_.dim);
  q_scaler_.resize(para_.dim);
  parameters_.resize(para_.number_of_variables);
  parameters_host_.resize(para_.number_of_variables);
  onsite_.resize(static_cast<std::size_t>(num_atoms_) * para_.num_onsite_outputs);
  hopping_.resize(static_cast<std::size_t>(num_bonds_) * para_.max_hopping_outputs);
  copy_to_gpu(para_.hopping_pair_offsets, hopping_pair_offsets_);

  grad_descriptors_.resize(static_cast<std::size_t>(num_atoms_) * para_.dim);
  grad_parameters_.resize(para_.number_of_variables);
  grad_onsite_.resize(static_cast<std::size_t>(num_atoms_) * para_.num_onsite_outputs);
  grad_hopping_.resize(static_cast<std::size_t>(num_bonds_) * para_.max_hopping_outputs);

  initialize_q_scaler();

  onsite_host_.resize(static_cast<std::size_t>(num_atoms_) * para_.num_onsite_outputs);
  hopping_host_.resize(static_cast<std::size_t>(num_bonds_) * para_.max_hopping_outputs);

  k_batch_size_ = std::min(para_.k_batch, num_kpoints_);
  eigensolver_.reset(new Eigensolver(num_orbitals_, k_batch_size_));
  precompute_sk_derivative_blocks();
}

TB::~TB() = default;

void TB::precompute_sk_derivative_blocks()
{
  const std::size_t block_size = static_cast<std::size_t>(orbitals_per_atom_) * orbitals_per_atom_;
  sk_derivative_blocks_.assign(
    static_cast<std::size_t>(num_bonds_) * para_.max_hopping_outputs * block_size, 0.0);
  active_hopping_outputs_.assign(num_bonds_, 0);

  for (int b = 0; b < num_bonds_; ++b) {
    const Neighbor& bond = dataset_.hopping_bonds[b];
    if (bond.r <= 0.0) {
      PRINT_INPUT_ERROR("Cannot precompute SK derivatives for a zero-length bond.");
    }
    const double l = bond.dx / bond.r;
    const double m = bond.dy / bond.r;
    const double n = bond.dz / bond.r;
    const int ti = dataset_.structure.atoms[bond.i].type;
    const int tj = dataset_.structure.atoms[bond.j].type;
    const int outputs = para_.hopping_outputs_for_pair(ti, tj);
    active_hopping_outputs_[b] = outputs;

    for (int mu_model = 0; mu_model < outputs; ++mu_model) {
      std::array<float, 21> h{};
      h[mu_model] = 1.0f;
      const std::size_t dst =
        (static_cast<std::size_t>(b) * para_.max_hopping_outputs + mu_model) * block_size;
      make_block(orbitals_per_atom_, ti, tj, l, m, n, h.data(), sk_derivative_blocks_.data() + dst);
    }
  }
}

void TB::build_hamiltonians(
  const std::vector<int>& k_indices, std::vector<double>& hr, std::vector<double>& hi) const
{
  constexpr double TWO_PI = 6.2831853071795864769;
  if (k_indices.empty()) {
    PRINT_INPUT_ERROR("Cannot build a zero-size k-point batch.");
  }
  for (int k : k_indices) {
    if (k < 0 || k >= num_kpoints_) {
      PRINT_INPUT_ERROR("k-point batch contains an out-of-range index.");
    }
  }

  const int batch_size = static_cast<int>(k_indices.size());
  const std::size_t matrix_size = static_cast<std::size_t>(num_orbitals_) * num_orbitals_;
  hr.assign(batch_size * matrix_size, 0.0);
  hi.assign(batch_size * matrix_size, 0.0);

  for (int kb = 0; kb < batch_size; ++kb) {
    for (int i = 0; i < num_atoms_; ++i) {
      const int o = orbitals_per_atom_ * i;
      const int p = para_.num_onsite_outputs * i;
      const double eps_s = onsite_host_[p];
      const double eps_p = onsite_host_[p + 1];
      hr[matrix_index(kb, o, o, num_orbitals_)] += eps_s;
      for (int a = 1; a <= 3; ++a) {
        hr[matrix_index(kb, o + a, o + a, num_orbitals_)] += eps_p;
      }
      if (para_.has_d_shell()) {
        const double eps_d = onsite_host_[p + 2];
        for (int a = 4; a <= 8; ++a) {
          hr[matrix_index(kb, o + a, o + a, num_orbitals_)] += eps_d;
        }
      }
      if (para_.has_sstar_shell()) {
        const int sstar = o + orbitals_per_atom_ - 1;
        hr[matrix_index(kb, sstar, sstar, num_orbitals_)] +=
          onsite_host_[p + para_.num_onsite_outputs - 1];
      }
    }
  }

  for (int b = 0; b < num_bonds_; ++b) {
    const Neighbor& bond = dataset_.hopping_bonds[b];
    if (bond.r <= 0.0) {
      PRINT_INPUT_ERROR("Encountered a zero-length hopping bond.");
    }

    const double l = bond.dx / bond.r;
    const double m = bond.dy / bond.r;
    const double n = bond.dz / bond.r;

    const int ti = dataset_.structure.atoms[bond.i].type;
    const int tj = dataset_.structure.atoms[bond.j].type;
    const std::size_t hop0 = static_cast<std::size_t>(para_.max_hopping_outputs) * b;
    std::array<double, 100> block{};
    make_block(orbitals_per_atom_, ti, tj, l, m, n, hopping_host_.data() + hop0, block.data());

    const int oi = orbitals_per_atom_ * bond.i;
    const int oj = orbitals_per_atom_ * bond.j;

    for (int kb = 0; kb < batch_size; ++kb) {
      const auto& kp = dataset_.kpoints[k_indices[kb]];
      const double k_dot_s = kp[0] * bond.sx + kp[1] * bond.sy + kp[2] * bond.sz;
      const double angle = TWO_PI * k_dot_s;
      const double phase_r = std::cos(angle);
      const double phase_i = -std::sin(angle);

      for (int a = 0; a < orbitals_per_atom_; ++a) {
        for (int c = 0; c < orbitals_per_atom_; ++c) {
          const double value = block[orbitals_per_atom_ * a + c];

          const std::size_t ij = matrix_index(kb, oi + a, oj + c, num_orbitals_);
          hr[ij] += value * phase_r;
          hi[ij] += value * phase_i;

          const std::size_t ji = matrix_index(kb, oj + c, oi + a, num_orbitals_);
          hr[ji] += value * phase_r;
          hi[ji] -= value * phase_i;
        }
      }
    }
  }
}

void TB::evaluate_bands(
  const double* parameters,
  const std::vector<int>& k_indices,
  std::vector<float>& eigenvalues,
  const bool with_vectors)
{
  compute_model(parameters);
  solve_bands(k_indices, eigenvalues, with_vectors);
}

void TB::solve_bands(
  const std::vector<int>& k_indices, std::vector<float>& eigenvalues, const bool with_vectors)
{
  if (static_cast<int>(k_indices.size()) != k_batch_size_) {
    PRINT_INPUT_ERROR(
      "Training k-point batch size does not match the persistent eigensolver size.");
  }
  std::vector<double> hr, hi;
  build_hamiltonians(k_indices, hr, hi);
  const std::size_t eval_size = static_cast<std::size_t>(k_batch_size_) * num_orbitals_;
  eigenvalues.resize(eval_size);
  eigenvectors_real_.resize(with_vectors ? eval_size * num_orbitals_ : 0);
  eigenvectors_imag_.resize(eigenvectors_real_.size());
  eigensolver_->solve(
    with_vectors,
    hr.data(),
    hi.data(),
    eigenvalues.data(),
    eigenvectors_real_.data(),
    eigenvectors_imag_.data());
}

void TB::backward(
  const std::vector<int>& k_indices,
  const std::vector<double>& eigenvalue_gradients,
  std::vector<float>& grad_parameters)
{
  constexpr double TWO_PI = 6.2831853071795864769;
  const std::size_t matrix_size = static_cast<std::size_t>(num_orbitals_) * num_orbitals_;
  const std::size_t expected_vec = k_batch_size_ * matrix_size;
  const std::size_t expected_eval = static_cast<std::size_t>(k_batch_size_) * num_orbitals_;
  if (
    static_cast<int>(k_indices.size()) != k_batch_size_ ||
    eigenvectors_real_.size() != expected_vec || eigenvectors_imag_.size() != expected_vec ||
    eigenvalue_gradients.size() != expected_eval) {
    PRINT_INPUT_ERROR("TB backward received inconsistent eigensystem dimensions.");
  }

  std::vector<double> grad_onsite, grad_hopping;
  grad_onsite.assign(static_cast<std::size_t>(num_atoms_) * para_.num_onsite_outputs, 0.0);
  grad_hopping.assign(static_cast<std::size_t>(num_bonds_) * para_.max_hopping_outputs, 0.0);

  std::vector<double> gr(expected_vec, 0.0);
  std::vector<double> gi(expected_vec, 0.0);
  for (int kb = 0; kb < k_batch_size_; ++kb) {
    const std::size_t vbase = kb * matrix_size;
    const std::size_t ebase = static_cast<std::size_t>(kb) * num_orbitals_;
    for (int row = 0; row < num_orbitals_; ++row) {
      for (int col = 0; col < num_orbitals_; ++col) {
        double real_sum = 0.0;
        double imag_sum = 0.0;
        for (int band = 0; band < num_orbitals_; ++band) {
          const double g = eigenvalue_gradients[ebase + band];
          if (g == 0.0) {
            continue;
          }
          const std::size_t ur = vbase + static_cast<std::size_t>(band) * num_orbitals_ + row;
          const std::size_t uc = vbase + static_cast<std::size_t>(band) * num_orbitals_ + col;
          const double ar = eigenvectors_real_[ur], ai = eigenvectors_imag_[ur];
          const double br = eigenvectors_real_[uc], bi = eigenvectors_imag_[uc];
          real_sum += g * (ar * br + ai * bi);
          imag_sum += g * (ai * br - ar * bi);
        }
        const std::size_t gc = vbase + static_cast<std::size_t>(col) * num_orbitals_ + row;
        gr[gc] = real_sum;
        gi[gc] = imag_sum;
      }
    }
  }

  for (int i = 0; i < num_atoms_; ++i) {
    const int o = orbitals_per_atom_ * i;
    const int p = para_.num_onsite_outputs * i;
    for (int kb = 0; kb < k_batch_size_; ++kb) {
      const std::size_t gbase = kb * matrix_size;
      auto diag = [&](int orbital) -> double {
        return gr[gbase + static_cast<std::size_t>(orbital) * num_orbitals_ + orbital];
      };
      grad_onsite[p] += diag(o);
      grad_onsite[p + 1] += diag(o + 1) + diag(o + 2) + diag(o + 3);
      if (para_.has_d_shell()) {
        for (int a = 4; a <= 8; ++a) {
          grad_onsite[p + 2] += diag(o + a);
        }
      }
      if (para_.has_sstar_shell()) {
        grad_onsite[p + para_.num_onsite_outputs - 1] += diag(o + orbitals_per_atom_ - 1);
      }
    }
  }

  const std::size_t local_block_size =
    static_cast<std::size_t>(orbitals_per_atom_) * orbitals_per_atom_;
  for (int b = 0; b < num_bonds_; ++b) {
    const Neighbor& bond = dataset_.hopping_bonds[b];
    const int oi = orbitals_per_atom_ * bond.i;
    const int oj = orbitals_per_atom_ * bond.j;
    const int outputs = active_hopping_outputs_[b];
    for (int kb = 0; kb < k_batch_size_; ++kb) {
      const auto& kp = dataset_.kpoints[k_indices[kb]];
      const double angle = TWO_PI * (kp[0] * bond.sx + kp[1] * bond.sy + kp[2] * bond.sz);
      const double pr = std::cos(angle);
      const double pi = -std::sin(angle);
      const std::size_t gbase = kb * matrix_size;

      for (int mu = 0; mu < outputs; ++mu) {
        const std::size_t db =
          (static_cast<std::size_t>(b) * para_.max_hopping_outputs + mu) * local_block_size;
        double value = 0.0;
        for (int a = 0; a < orbitals_per_atom_; ++a) {
          for (int c = 0; c < orbitals_per_atom_; ++c) {
            const double jac =
              sk_derivative_blocks_[db + static_cast<std::size_t>(a) * orbitals_per_atom_ + c];
            if (jac == 0.0) {
              continue;
            }
            const std::size_t gc =
              gbase + static_cast<std::size_t>(oi + a) * num_orbitals_ + (oj + c);
            value += 2.0 * jac * (gr[gc] * pr - gi[gc] * pi);
          }
        }
        grad_hopping[static_cast<std::size_t>(b) * para_.max_hopping_outputs + mu] += value;
      }
    }
  }

  copy_to_gpu(grad_onsite, grad_onsite_);
  copy_to_gpu(grad_hopping, grad_hopping_);
  grad_descriptors_.fill(0.0f);
  grad_parameters_.fill(0.0f);

  const int block_size = 128;
  const int atom_grid = (num_atoms_ + block_size - 1) / block_size;
  const int bond_grid = (num_bonds_ + block_size - 1) / block_size;

  backward_onsite_ann<<<atom_grid, block_size>>>(
    num_atoms_,
    onsite_meta(para_),
    atom_types_.data(),
    descriptors_.data(),
    q_scaler_.data(),
    parameters_.data(),
    grad_onsite_.data(),
    grad_descriptors_.data(),
    grad_parameters_.data());
  GPU_CHECK_KERNEL

  if (num_bonds_ > 0) {
    backward_hopping_ann<<<bond_grid, block_size>>>(
      num_bonds_,
      num_atoms_,
      hopping_meta(para_),
      atom_types_.data(),
      bond_i_.data(),
      bond_j_.data(),
      bond_r_.data(),
      descriptors_.data(),
      q_scaler_.data(),
      parameters_.data(),
      hopping_pair_offsets_.data(),
      grad_hopping_.data(),
      grad_descriptors_.data(),
      grad_parameters_.data());
    GPU_CHECK_KERNEL
  }

  const DescriptorMeta meta = descriptor_meta(para_);
  backward_descriptor_radial<<<atom_grid, block_size>>>(
    num_atoms_,
    meta,
    NeighborView{
      radial_.nn_sum_.data(),
      radial_.nn_.data(),
      radial_.nl_.data(),
      radial_.dx_.data(),
      radial_.dy_.data(),
      radial_.dz_.data()},
    atom_types_.data(),
    grad_descriptors_.data(),
    grad_parameters_.data() + para_.offset_descriptor);
  GPU_CHECK_KERNEL

  backward_descriptor_angular<<<atom_grid, block_size>>>(
    num_atoms_,
    meta,
    NeighborView{
      angular_.nn_sum_.data(),
      angular_.nn_.data(),
      angular_.nl_.data(),
      angular_.dx_.data(),
      angular_.dy_.data(),
      angular_.dz_.data()},
    atom_types_.data(),
    parameters_.data() + para_.offset_descriptor,
    grad_descriptors_.data(),
    grad_parameters_.data() + para_.offset_descriptor);
  GPU_CHECK_KERNEL

  grad_parameters.resize(para_.number_of_variables);
  grad_parameters_.copy_to_host(grad_parameters.data());
}

void TB::evaluate_all_bands(const double* parameters, std::vector<float>& eigenvalues)
{
  compute_model(parameters);
  solve_all_bands(eigenvalues);
}

void TB::solve_all_bands(std::vector<float>& eigenvalues, Result* result)
{
  eigenvalues.assign(static_cast<std::size_t>(num_kpoints_) * num_orbitals_, 0.0f);
  if (result) {
    result->eigenvectors_real.resize(eigenvalues.size() * num_orbitals_);
    result->eigenvectors_imag.resize(result->eigenvectors_real.size());
  }
  std::vector<float> batch_eigenvalues;
  std::vector<int> k_indices(k_batch_size_);

  for (int start = 0; start < num_kpoints_; start += k_batch_size_) {
    const int valid = std::min(k_batch_size_, num_kpoints_ - start);
    for (int t = 0; t < k_batch_size_; ++t) {
      k_indices[t] = (start + t) % num_kpoints_;
    }

    solve_bands(k_indices, batch_eigenvalues, result != nullptr);

    for (int t = 0; t < valid; ++t) {
      const std::size_t src = static_cast<std::size_t>(t) * num_orbitals_;
      const std::size_t dst = static_cast<std::size_t>(start + t) * num_orbitals_;
      std::copy(
        batch_eigenvalues.begin() + src,
        batch_eigenvalues.begin() + src + num_orbitals_,
        eigenvalues.begin() + dst);
      if (result) {
        const std::size_t begin = src * num_orbitals_;
        const std::size_t end = (src + num_orbitals_) * num_orbitals_;
        std::copy(
          eigenvectors_real_.begin() + begin,
          eigenvectors_real_.begin() + end,
          result->eigenvectors_real.begin() + dst * num_orbitals_);
        std::copy(
          eigenvectors_imag_.begin() + begin,
          eigenvectors_imag_.begin() + end,
          result->eigenvectors_imag.begin() + dst * num_orbitals_);
      }
    }
  }
  eigenvectors_real_.clear();
  eigenvectors_imag_.clear();
}

void TB::forward(const double* parameters, TB::Result& result)
{
  compute_model(parameters);
  result.descriptors.resize(static_cast<std::size_t>(num_atoms_) * para_.dim);
  descriptors_.copy_to_host(result.descriptors.data());
  result.onsite = onsite_host_;
  result.hopping = hopping_host_;
  result.num_orbitals = num_orbitals_;
  result.num_kpoints = num_kpoints_;

  std::vector<int> all_kpoints(num_kpoints_);
  std::iota(all_kpoints.begin(), all_kpoints.end(), 0);
  build_hamiltonians(all_kpoints, result.hamiltonian_real, result.hamiltonian_imag);

  solve_all_bands(result.eigenvalues, &result);
}
