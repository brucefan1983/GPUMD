/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

#include "nep_extrapolation.cuh"
#include "utilities/nep_utilities.cuh"

static __device__ void apply_mic_small_box(
  const Box& box, const NEP_Extrapolation::ExpandedBox& ebox, float& x12, float& y12, float& z12)
{
  float sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
  float sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
  float sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
  if (box.pbc_x == 1)
    sx12 -= nearbyint(sx12);
  if (box.pbc_y == 1)
    sy12 -= nearbyint(sy12);
  if (box.pbc_z == 1)
    sz12 -= nearbyint(sz12);
  x12 = ebox.h[0] * sx12 + ebox.h[1] * sy12 + ebox.h[2] * sz12;
  y12 = ebox.h[3] * sx12 + ebox.h[4] * sy12 + ebox.h[5] * sz12;
  z12 = ebox.h[6] * sx12 + ebox.h[7] * sy12 + ebox.h[8] * sz12;
}

static __global__ void find_neighbor_list_small_box(
  NEP_Extrapolation::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const NEP_Extrapolation::ExpandedBox ebox,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
        for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
          for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }

            float delta[3];
            delta[0] = box.float_h[0] * ia + box.float_h[1] * ib + box.float_h[2] * ic;
            delta[1] = box.float_h[3] * ia + box.float_h[4] * ib + box.float_h[5] * ic;
            delta[2] = box.float_h[6] * ia + box.float_h[7] * ib + box.float_h[8] * ic;

            float x12 = g_x[n2] + delta[0] - x1;
            float y12 = g_y[n2] + delta[1] - y1;
            float z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(box, ebox, x12, y12, z12);

            float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);

            int t2 = g_type[n2];
            float rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
            float rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;

            if (distance_square < rc_radial * rc_radial) {
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = x12;
              g_y12_radial[count_radial * N + n1] = y12;
              g_z12_radial[count_radial * N + n1] = z12;
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = x12;
              g_y12_angular[count_angular * N + n1] = y12;
              g_z12_angular[count_angular * N + n1] = z12;
              count_angular++;
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

static __global__ void find_descriptor_small_box(
  NEP_Extrapolation::ParaMB paramb,
  NEP_Extrapolation::ANN annmb,
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
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz,
  bool need_B_projection,
  double* B_projection,
  int B_projection_size)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (t1 * paramb.num_types + t2) * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1));
          c_index += n * (paramb.basis_size_radial + 1) + k;
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = paramb.num_c_radial;
          c_index += (t1 * paramb.num_types + t2) * ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
          c_index += n * (paramb.basis_size_angular + 1) + k;
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, r12[0], r12[1], r12[2], gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (!need_B_projection)
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp);
    else
      apply_ann_one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        B_projection + n1 * B_projection_size);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}
