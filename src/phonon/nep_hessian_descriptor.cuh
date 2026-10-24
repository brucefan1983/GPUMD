/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
NEP descriptor-only kernel for the analytic Hessian.

The descriptor-computation logic in src/force/nep.cu is a `static` kernel
that cannot be invoked from another translation unit, and the side-channel
write of q_descriptors that the hessian-only branch added in that file is
no longer present after the hessian-only branch is rewritten to make no
modifications to existing source files.  This header therefore provides a
self-contained, descriptor-only kernel that the hessian module owns and
calls to populate its own q cache.

The kernel mirrors the descriptor portion of nep.cu's find_descriptor
(untouched in src/force/nep.cu).  The output is the normalized
per-atom q array, written in the same SoA layout as
nep_data.q_descriptors had: dim-major (descriptor component stride N).
------------------------------------------------------------------------------*/

#pragma once
#include "force/nep.cuh"
#include "model/box.cuh"
#include "utilities/nep_utilities.cuh"

#ifndef HESSIAN_DESCRIPTOR_MAX_DIM
#define HESSIAN_DESCRIPTOR_MAX_DIM 256
#endif

namespace nep_hessian {

struct DescriptorCache {
  GPU_Vector<float> q; // dim * N, SoA, dim-major
  int last_num_atoms = 0;
  int last_dim = 0;
};

__global__ void find_descriptor_only(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  float* g_q_desc)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  int t1 = g_type[n1];
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  float q[HESSIAN_DESCRIPTOR_MAX_DIM] = {0.0f};

  for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
    int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    int t2 = g_type[n2];
    float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index = get_c_index(
          t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
        gn12 += fn12[k] * annmb.c_type_pair[c_index];
      }
      q[n] += gn12;
    }
  }

  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = get_c_index(
          t1 * paramb.num_types + t2,
          n,
          k,
          paramb.n_max_angular,
          paramb.basis_size_angular,
          paramb.num_c_radial);
        gn12 += fn12[k] * annmb.c_type_pair[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }
    find_q(
      paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
      paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n,
      s, q + (paramb.n_max_radial + 1));
  }

  for (int d = 0; d < annmb.dim; ++d) {
    q[d] = q[d] * annmb.q_scaler[d];
    g_q_desc[static_cast<size_t>(d) * N + n1] = q[d];
  }
}

// Small-box variant: uses pre-computed r12 displacements (with MIC already
// applied) and the small-box neighbor layout, so it produces q values that
// are bit-identical to those written by find_descriptor_small_box.
__global__ void find_descriptor_only_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  float* g_q_desc)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;
  int t1 = g_type[n1];
  float q[HESSIAN_DESCRIPTOR_MAX_DIM] = {0.0f};

  for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
    int index = i1 * N + n1;
    int n2 = g_NL_radial[index];
    float x12 = g_x12_radial[index];
    float y12 = g_y12_radial[index];
    float z12 = g_z12_radial[index];
    float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    int t2 = g_type[n2];
    float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        int c_index = get_c_index(
          t1 * paramb.num_types + t2,
          n,
          k,
          paramb.n_max_radial,
          paramb.basis_size_radial);
        gn12 += fn12[k] * annmb.c_type_pair[c_index];
      }
      q[n] += gn12;
    }
  }

  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      float x12 = g_x12_angular[index];
      float y12 = g_y12_angular[index];
      float z12 = g_z12_angular[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = get_c_index(
          t1 * paramb.num_types + t2,
          n,
          k,
          paramb.n_max_angular,
          paramb.basis_size_angular,
          paramb.num_c_radial);
        gn12 += fn12[k] * annmb.c_type_pair[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }
    find_q(
      paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123,
      paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n,
      s, q + (paramb.n_max_radial + 1));
  }

  for (int d = 0; d < annmb.dim; ++d) {
    q[d] = q[d] * annmb.q_scaler[d];
    g_q_desc[static_cast<size_t>(d) * N + n1] = q[d];
  }
}

} // namespace nep_hessian
