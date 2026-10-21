/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/

#pragma once

// Adapted from NEPAdapters' unified nep4_spin3 O/C CUDA implementation.
__device__ __forceinline__ float spin3_dot3(
    const float* left,
    const float* right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

__device__ __forceinline__ void spin3_cross3(
    const float* left,
    const float* right,
    float* out) {
  out[0] = left[1] * right[2] - left[2] * right[1];
  out[1] = left[2] * right[0] - left[0] * right[2];
  out[2] = left[0] * right[1] - left[1] * right[0];
}

__device__ __forceinline__ void spin3_add_cross_pull(
    const float* left,
    const float* right,
    const float* pull,
    float* pull_left,
    float* pull_right) {
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

__device__ __forceinline__ int spin3_oc_channel_offset(int channel) {
  return channel * kSpin3OcDensityStride;
}

__device__ __forceinline__ int spin3_oc_same_offset(int channels) {
  return channels * kSpin3OcDensityStride;
}

__device__ __forceinline__ int spin3_oc_pair_index(
    int channels,
    int left,
    int right) {
  return left * channels - left * (left - 1) / 2 + right - left;
}

__device__ __forceinline__ float spin3_oc_mix(
    const float* projection,
    int channels,
    int leg,
    int row,
    int source) {
  // Torch flatten order for radial_leg_mix[4][C][C] is
  // [leg][output row][input/source channel].
  return projection[(leg * channels + row) * channels + source];
}
