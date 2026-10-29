/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/

#pragma once

// Adapted from NEPAdapters' unified nep4_spin3 O/C CUDA implementation.
__device__ __forceinline__ float spin3_dot3(const float* left, const float* right)
{
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

__device__ __forceinline__ void spin3_cross3(const float* left, const float* right, float* out)
{
  out[0] = left[1] * right[2] - left[2] * right[1];
  out[1] = left[2] * right[0] - left[0] * right[2];
  out[2] = left[0] * right[1] - left[1] * right[0];
}

__device__ __forceinline__ void spin3_add_cross_pull(
  const float* left, const float* right, const float* pull, float* pull_left, float* pull_right)
{
  float right_cross_pull[3], pull_cross_left[3];
  spin3_cross3(right, pull, right_cross_pull);
  spin3_cross3(pull, left, pull_cross_left);
  for (int d = 0; d < 3; ++d) {
    pull_left[d] += right_cross_pull[d];
    pull_right[d] += pull_cross_left[d];
  }
}

// Canonical center state for the unified nep4_spin3 O/C descriptor.
// linear contractions of these moments (W, DW, QS and the edge-anchored
// L11/L22/L112 responses) are reconstructed after the neighbor reduction.
// This keeps the edge traversal and analytical VJP on the minimum closed
// moment set instead of storing several equivalent representations.
enum Spin3OcDensityOffset : int {
  kSpin3OcM = 0,
  kSpin3OcP = 3,
  kSpin3OcL = 6,
  kSpin3OcX = 7,
  kSpin3OcT = 10,
  kSpin3OcQ = 15,
  kSpin3OcQP = 20,
  kSpin3OcDM = 35,
  kSpin3OcDensityStride = 38,
};

__device__ __forceinline__ int spin3_oc_channel_offset(int channel)
{
  return channel * kSpin3OcDensityStride;
}

__device__ __forceinline__ int spin3_oc_same_offset(int channels)
{
  return channels * kSpin3OcDensityStride;
}

__device__ __forceinline__ int spin3_oc_pair_index(int channels, int left, int right)
{
  return left * channels - left * (left - 1) / 2 + right - left;
}

__device__ __forceinline__ float
spin3_oc_mix(const float* projection, int channels, int leg, int row, int source)
{
  // Torch flatten order for radial_leg_mix[4][C][C] is
  // [leg][output row][input/source channel].
  return projection[(leg * channels + row) * channels + source];
}

// Shared descriptor and moment indexing for training and MD.
template <typename Layout>
Layout
make_spin_polynomial_layout(const int channels, const int l_max, const int order, const int soc)
{
  Layout layout;
  layout.channels = channels;
  layout.pair_count = channels * (channels + 1) / 2;
  layout.moment_count = layout.density_stride * channels + layout.pair_count;
  if (order >= 2 && l_max >= 1) {
    layout.angular_l1_moment_offset = layout.moment_count;
    layout.moment_count += 3 * channels;
  }
  if (order >= 2 && l_max >= 2) {
    layout.angular_l2_moment_offset = layout.moment_count;
    layout.moment_count += 5 * channels;
  }
  int offset = 0;
  layout.local_s2 = offset++;
  layout.edge_l0_dot = offset;
  offset += channels;
  layout.edge_l0_neighbor_s2 = offset;
  offset += channels;
  if (soc != 0 && l_max >= 2) {
    layout.edge_l2_pair = offset;
    offset += channels;
    layout.center_l2_environment = offset;
    offset += channels;
  }
  if (order >= 2) {
    layout.edge_l0_dot2 = offset;
    offset += channels;
    layout.density_l0_self = offset;
    offset += channels;
    if (l_max >= 1) {
      if (soc != 0) {
        layout.density_l1_longitudinal_self = offset;
        offset += channels;
        layout.density_l1_axial_self = offset;
        offset += channels;
        layout.density_l1_stf_self = offset;
        offset += channels;
      } else {
        layout.density_l1_product_self = offset;
        offset += channels;
      }
    }
    if (l_max >= 2) {
      layout.density_l2_product_self = offset;
      offset += channels;
    }
    layout.density_l0_dot_response = offset;
    offset += channels;
    layout.correlation_same_edge = offset;
    offset += layout.pair_count;
    layout.correlation_distinct_neighbor = offset;
    offset += layout.pair_count;
    if (l_max >= 1) {
      layout.correlation_distinct_l1 = offset;
      offset += layout.pair_count;
    }
    if (l_max >= 2) {
      layout.correlation_distinct_l2 = offset;
      offset += layout.pair_count;
    }
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) {
        layout.coupling_l11_axial = offset;
        offset += channels;
      }
      layout.edge_l11_axial = offset;
      offset += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) {
        layout.coupling_l22_axial = offset;
        offset += channels;
      }
      layout.edge_l22_axial = offset;
      offset += channels;
    }
  }
  if (order >= 3) {
    layout.edge_l0_moment_gate = offset;
    offset += channels;
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) {
        layout.coupling_l11_dot_response = offset;
        offset += channels;
      }
      layout.coupling_l111_p_m_x = offset;
      offset += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) {
        layout.coupling_l22_dot_response = offset;
        offset += channels;
      }
      layout.coupling_l111_p_qs_x = offset;
      offset += channels;
      if (channels >= 2) {
        layout.coupling_l112_edge_response = offset;
        offset += channels;
      }
    }
    if (soc != 0 && l_max >= 1 && channels >= 3) {
      layout.coupling_l111_bulk = offset;
      offset += channels;
    }
  }
  layout.descriptor_dim = offset;
  return layout;
}
