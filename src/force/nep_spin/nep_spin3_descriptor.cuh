/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/

#pragma once

// Unified O/C spin3 descriptor kernels. Included from nep_spin.cu.
// Unified spin3 follows the Torch magnetic-polynomial STF5 ABI: five orthonormal
// components, not the historical spin3 compact-matrix coordinates.
__device__ __forceinline__ void spin3_oc_stf5_outer(
    const float* left,
    const float* right,
    float* out) {
  constexpr float InvSqrt6 = 0.40824829046386301637f;
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  const float xx = left[0] * right[0];
  const float yy = left[1] * right[1];
  const float zz = left[2] * right[2];
  out[0] = (2.0f * zz - xx - yy) * InvSqrt6;
  out[1] = (xx - yy) * InvSqrt2;
  out[2] = (left[0] * right[1] + left[1] * right[0]) * InvSqrt2;
  out[3] = (left[1] * right[2] + left[2] * right[1]) * InvSqrt2;
  out[4] = (left[2] * right[0] + left[0] * right[2]) * InvSqrt2;
}

__device__ __forceinline__ void spin3_oc_expand_stf5(
    const float* value,
    float* out) {
  constexpr float InvSqrt6 = 0.40824829046386301637f;
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  const float xx = -value[0] * InvSqrt6 + value[1] * InvSqrt2;
  const float yy = -value[0] * InvSqrt6 - value[1] * InvSqrt2;
  const float zz = 2.0f * value[0] * InvSqrt6;
  const float xy = value[2] * InvSqrt2;
  const float yz = value[3] * InvSqrt2;
  const float zx = value[4] * InvSqrt2;
  out[0] = xx; out[1] = xy; out[2] = zx;
  out[3] = xy; out[4] = yy; out[5] = yz;
  out[6] = zx; out[7] = yz; out[8] = zz;
}

__device__ __forceinline__ void spin3_oc_stf5_matvec(
    const float* matrix,
    const float* vector,
    float* out) {
  float full[9];
  spin3_oc_expand_stf5(matrix, full);
  for (int a = 0; a < 3; ++a) {
    out[a] = 0.0f;
    for (int b = 0; b < 3; ++b) {
      out[a] += full[3 * a + b] * vector[b];
    }
  }
}

__device__ __forceinline__ void spin3_oc_add_stf5_outer_pull(
    const float* left,
    const float* right,
    const float* pull,
    float scale,
    float* pull_left,
    float* pull_right) {
  constexpr float InvSqrt6 = 0.40824829046386301637f;
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  const float g0 = scale * pull[0];
  const float g1 = scale * pull[1];
  const float g2 = scale * pull[2];
  const float g3 = scale * pull[3];
  const float g4 = scale * pull[4];
#define NEP_SPIN3_OC_STF_SIDE(target, other)                                  \
  do {                                                                     \
    (target)[0] += (-g0 * InvSqrt6 + g1 * InvSqrt2) * (other)[0] +        \
        g2 * InvSqrt2 * (other)[1] + g4 * InvSqrt2 * (other)[2];          \
    (target)[1] += g2 * InvSqrt2 * (other)[0] +                            \
        (-g0 * InvSqrt6 - g1 * InvSqrt2) * (other)[1] +                   \
        g3 * InvSqrt2 * (other)[2];                                       \
    (target)[2] += g4 * InvSqrt2 * (other)[0] +                            \
        g3 * InvSqrt2 * (other)[1] + 2.0f * g0 * InvSqrt6 * (other)[2];   \
  } while (false)
  NEP_SPIN3_OC_STF_SIDE(pull_left, right);
  NEP_SPIN3_OC_STF_SIDE(pull_right, left);
#undef NEP_SPIN3_OC_STF_SIDE
}

__device__ __forceinline__ void spin3_oc_add_full_matrix_pull_to_stf5(
    const float* full,
    float* pull) {
  constexpr float InvSqrt6 = 0.40824829046386301637f;
  constexpr float InvSqrt2 = 0.70710678118654752440f;
  pull[0] += (-full[0] - full[4] + 2.0f * full[8]) * InvSqrt6;
  pull[1] += (full[0] - full[4]) * InvSqrt2;
  pull[2] += (full[1] + full[3]) * InvSqrt2;
  pull[3] += (full[5] + full[7]) * InvSqrt2;
  pull[4] += (full[2] + full[6]) * InvSqrt2;
}

__device__ __forceinline__ void spin3_oc_axial_commutator(
    const float* left,
    const float* right,
    float* out) {
  float l[9], r[9], comm[9] = {};
  spin3_oc_expand_stf5(left, l);
  spin3_oc_expand_stf5(right, r);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int k = 0; k < 3; ++k) {
        comm[3 * a + b] +=
            l[3 * a + k] * r[3 * k + b] -
            r[3 * a + k] * l[3 * k + b];
      }
    }
  }
  out[0] = comm[7];
  out[1] = comm[2];
  out[2] = comm[3];
}

__device__ __forceinline__ void spin3_oc_add_commutator_axial_pull(
    const float* left,
    const float* right,
    const float* pull,
    float* pull_left,
    float* pull_right) {
  float l[9], r[9], gl[9] = {}, gr[9] = {}, h[9] = {};
  spin3_oc_expand_stf5(left, l);
  spin3_oc_expand_stf5(right, r);
  h[7] = pull[0];
  h[2] = pull[1];
  h[3] = pull[2];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int k = 0; k < 3; ++k) {
        gl[3 * a + k] += h[3 * a + b] * r[3 * k + b];
        gr[3 * k + b] += h[3 * a + b] * l[3 * a + k];
        gr[3 * a + k] -= h[3 * a + b] * l[3 * k + b];
        gl[3 * k + b] -= h[3 * a + b] * r[3 * a + k];
      }
    }
  }
  spin3_oc_add_full_matrix_pull_to_stf5(gl, pull_left);
  spin3_oc_add_full_matrix_pull_to_stf5(gr, pull_right);
}

template <int Width>
__device__ __forceinline__ void spin3_oc_project_density(
    const float* __restrict__ center,
    const float* __restrict__ projection,
    int channels,
    int density_offset,
    int leg,
    int row,
    float* __restrict__ out) {
  for (int k = 0; k < Width; ++k) {
    out[k] = 0.0f;
    for (int source = 0; source < channels; ++source) {
      out[k] += spin3_oc_mix(projection, channels, leg, row, source) *
          center[spin3_oc_channel_offset(source) + density_offset + k];
    }
  }
}

template <int Width>
__device__ __forceinline__ float spin3_oc_dotn(
    const float* left,
    const float* right) {
  float value = 0.0f;
  for (int k = 0; k < Width; ++k) {
    value += left[k] * right[k];
  }
  return value;
}

__device__ __forceinline__ void spin3_oc_qp_matrix(
    const float* qp,
    int spin_component,
    float* matrix) {
  float coefficients[5];
  for (int k = 0; k < 5; ++k) {
    coefficients[k] = qp[3 * k + spin_component];
  }
  spin3_oc_expand_stf5(coefficients, matrix);
}

__device__ __forceinline__ void spin3_oc_qp_vector(
    const float* qp,
    float* vector) {
  vector[0] = 0.0f;
  vector[1] = 0.0f;
  vector[2] = 0.0f;
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    float matrix[9];
    spin3_oc_qp_matrix(qp, spin_component, matrix);
    for (int a = 0; a < 3; ++a) {
      vector[a] += matrix[3 * a + spin_component];
    }
  }
}

__device__ __forceinline__ void spin3_oc_l11_axis_from_canonical(
    const float* si,
    float longitudinal,
    const float* axial,
    const float* stf_first,
    float* axis) {
  float matrix[9], matrix_on_spin[3], axial_cross_spin[3];
  spin3_oc_expand_stf5(stf_first, matrix);
  for (int a = 0; a < 3; ++a) {
    matrix_on_spin[a] = matrix[3 * a] * si[0] +
        matrix[3 * a + 1] * si[1] + matrix[3 * a + 2] * si[2];
  }
  spin3_cross3(axial, si, axial_cross_spin);
  for (int d = 0; d < 3; ++d) {
    axis[d] = (2.0f / 3.0f) * longitudinal * si[d] -
        matrix_on_spin[d] - 0.5f * axial_cross_spin[d];
  }
}

__device__ __forceinline__ void spin3_oc_qp_raw_matrix(
    const float* moment,
    const float* qp,
    int spin_component,
    float* matrix) {
  spin3_oc_qp_matrix(qp, spin_component, matrix);
  for (int a = 0; a < 3; ++a) {
    matrix[3 * a + a] += moment[spin_component] / 3.0f;
  }
}

__device__ __forceinline__ float spin3_oc_l22_scalar_from_canonical(
    const float* si,
    const float* moment,
    const float* projected_q,
    const float* projected_qp) {
  float q_matrix[9], q_on_spin[3];
  spin3_oc_expand_stf5(projected_q, q_matrix);
  for (int a = 0; a < 3; ++a) {
    q_on_spin[a] = 0.0f;
    for (int b = 0; b < 3; ++b) {
      q_on_spin[a] += q_matrix[3 * a + b] * si[b];
    }
  }
  float value = 0.0f;
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    float raw_second[9];
    spin3_oc_qp_raw_matrix(
        moment, projected_qp, spin_component, raw_second);
    for (int a = 0; a < 3; ++a) {
      value -= raw_second[3 * spin_component + a] * q_on_spin[a];
      for (int b = 0; b < 3; ++b) {
        value += q_matrix[3 * spin_component + a] *
            raw_second[3 * a + b] * si[b];
      }
    }
  }
  return value;
}


template <int C>
__global__ void build_spin3_oc_density_bank(
    SpinPolynomialLayout layout,
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_basis_size,
    float spin_cutoff,
    const float* __restrict__ spin_cutoff_pair,
    int spin_coefficient_offset,
    SimulationBox box,
    const int* __restrict__ types,
    const int* __restrict__ spin_type_active,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ slot_r12,
    int r12_plane_size,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial_slot_major,
    const float* __restrict__ descriptor_coefficients,
    float* __restrict__ descriptors,
    float* __restrict__ moments) {
  const int work = blockIdx.x * blockDim.x + threadIdx.x;
  const int atom = work / C;
  const int channel = work - atom * C;
  if (atom >= atom_count) return;

  float density[kSpin3OcDensityStride] = {};
  float same[C * (C + 1) / 2] = {};
  float edge_sj2 = 0.0f;
  float edge_dot2 = 0.0f;
  float edge_gate = 0.0f;
  float angular_l1[3] = {};
  float angular_l2[5] = {};
  const bool active = spin_type_active[types[atom]] != 0;
  if (active) {
    const int neighbor_count = nn_radial[atom];
    for (int slot = 0; slot < neighbor_count; ++slot) {
      const int neighbor = nl_radial_slot_major[slot * atom_stride + atom];
      if (neighbor < 0) continue;
      float r[3], dist, si[3], sj[3], weights[C], derivatives[C];
      if (!load_spin3_edge_f32<C, false>(
              atom, neighbor, atom_stride, num_types, spin_basis_size,
              spin_cutoff, spin_cutoff_pair, box, types, positions_soa3, slot_r12,
              r12_plane_size, slot * atom_stride + atom, spins_soa3,
              descriptor_coefficients, spin_coefficient_offset, r, dist, si,
              sj, weights, derivatives)) continue;
      const float weight = weights[channel];
      const float si2 = spin3_dot3(si, si);
      const float sj2 = spin3_dot3(sj, sj);
      const float dot = spin3_dot3(si, sj);
      const float longitudinal = spin3_dot3(r, sj);
      float axial[3], qrr[5], edge_stf[5];
      spin3_cross3(r, sj, axial);
      spin3_oc_stf5_outer(r, r, qrr);
      spin3_oc_stf5_outer(r, sj, edge_stf);
      for (int d = 0; d < 3; ++d) {
        density[kSpin3OcM + d] += weight * sj[d];
        density[kSpin3OcP + d] += weight * r[d];
        density[kSpin3OcX + d] += weight * axial[d];
        density[kSpin3OcDM + d] += weight * dot * sj[d];
        if (layout.angular_l1_moment_offset >= 0) {
          angular_l1[d] += weight * dot * r[d];
        }
      }
      density[kSpin3OcL] += weight * longitudinal;
      for (int k = 0; k < 5; ++k) {
        density[kSpin3OcT + k] += weight * edge_stf[k];
        density[kSpin3OcQ + k] += weight * qrr[k];
        if (layout.angular_l2_moment_offset >= 0) {
          angular_l2[k] += weight * dot * qrr[k];
        }
        for (int d = 0; d < 3; ++d) {
          density[kSpin3OcQP + 3 * k + d] += weight * qrr[k] * sj[d];
        }
      }
      edge_sj2 += weight * sj2;
      edge_dot2 += weight * dot * dot;
      edge_gate += weight * dot * (si2 + sj2);
      if (channel == 0 && layout.correlation_same_edge >= 0) {
        for (int left = 0; left < C; ++left) {
          for (int right = left; right < C; ++right) {
            same[spin3_oc_pair_index(C, left, right)] +=
                weights[left] * weights[right] * dot * dot;
          }
        }
      }
    }
  }
  float* center = moments + atom * layout.moment_count;
  const int base = spin3_oc_channel_offset(channel);
  for (int k = 0; k < kSpin3OcDensityStride; ++k) center[base + k] = density[k];
  if (layout.angular_l1_moment_offset >= 0) {
    for (int d = 0; d < 3; ++d) {
      center[layout.angular_l1_moment_offset + 3 * channel + d] = angular_l1[d];
    }
  }
  if (layout.angular_l2_moment_offset >= 0) {
    for (int k = 0; k < 5; ++k) {
      center[layout.angular_l2_moment_offset + 5 * channel + k] = angular_l2[k];
    }
  }
  if (channel == 0) {
    const int same_offset = spin3_oc_same_offset(C);
    for (int pair = 0; pair < layout.pair_count; ++pair) {
      center[same_offset + pair] = same[pair];
    }
  }
#define NEP_SPIN3_OC_STORE_DYNAMIC(index, value)                         \
  do { if ((index) >= 0) descriptors[atom + atom_stride *               \
      (struct_dim + (index) + channel)] = active ? (value) : 0.0f; } while (false)
  NEP_SPIN3_OC_STORE_DYNAMIC(layout.edge_l0_neighbor_s2, edge_sj2);
  NEP_SPIN3_OC_STORE_DYNAMIC(layout.edge_l0_dot2, edge_dot2);
  NEP_SPIN3_OC_STORE_DYNAMIC(layout.edge_l0_moment_gate, edge_gate);
#undef NEP_SPIN3_OC_STORE_DYNAMIC
}

template <int C>
__global__ void contract_spin3_oc_descriptors(
    SpinPolynomialLayout layout,
    int atom_count,
    int atom_stride,
    int struct_dim,
    const int* __restrict__ types,
    const int* __restrict__ spin_type_active,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ projection,
    const float* __restrict__ moments,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) return;
  const bool active = spin_type_active[types[atom]] != 0;
  const float* center = moments + atom * layout.moment_count;
  float si[3] = {
      static_cast<float>(spins_soa3[atom]),
      static_cast<float>(spins_soa3[atom_stride + atom]),
      static_cast<float>(spins_soa3[2 * atom_stride + atom])};
#define NEP_SPIN3_OC_STORE_AT(index, value)                              \
  do { if ((index) >= 0) descriptors[atom + atom_stride *               \
      (struct_dim + (index))] = active ? (value) : 0.0f; } while (false)
  NEP_SPIN3_OC_STORE_AT(layout.local_s2, spin3_dot3(si, si));
  float local_q[5];
  float first_dot[C];
  spin3_oc_stf5_outer(si, si, local_q);
  for (int c = 0; c < C; ++c) {
    const int base = spin3_oc_channel_offset(c);
    const float* m = center + base + kSpin3OcM;
    first_dot[c] = spin3_dot3(si, m);
    NEP_SPIN3_OC_STORE_AT(layout.edge_l0_dot + c, first_dot[c]);
    if (layout.edge_l2_pair >= 0) {
      float q_on_spin[3];
      spin3_oc_qp_vector(center + base + kSpin3OcQP, q_on_spin);
      NEP_SPIN3_OC_STORE_AT(
          layout.edge_l2_pair + c, spin3_dot3(si, q_on_spin));
      NEP_SPIN3_OC_STORE_AT(
          layout.center_l2_environment + c,
          spin3_oc_dotn<5>(center + base + kSpin3OcQ, local_q));
    }
    if (layout.density_l0_self >= 0) {
      NEP_SPIN3_OC_STORE_AT(
          layout.density_l0_self + c, spin3_oc_dotn<3>(m, m));
      if (layout.density_l1_longitudinal_self >= 0) {
        NEP_SPIN3_OC_STORE_AT(
            layout.density_l1_longitudinal_self + c,
            center[base + kSpin3OcL] * center[base + kSpin3OcL]);
        NEP_SPIN3_OC_STORE_AT(
            layout.density_l1_axial_self + c,
            spin3_oc_dotn<3>(center + base + kSpin3OcX,
                             center + base + kSpin3OcX));
        NEP_SPIN3_OC_STORE_AT(
            layout.density_l1_stf_self + c,
            spin3_oc_dotn<5>(center + base + kSpin3OcT,
                             center + base + kSpin3OcT));
      }
      if (layout.density_l1_product_self >= 0) {
        const float raw_norm =
            center[base + kSpin3OcL] * center[base + kSpin3OcL] / 3.0f +
            spin3_oc_dotn<3>(center + base + kSpin3OcX,
                             center + base + kSpin3OcX) * 0.5f +
            spin3_oc_dotn<5>(center + base + kSpin3OcT,
                             center + base + kSpin3OcT);
        NEP_SPIN3_OC_STORE_AT(layout.density_l1_product_self + c, raw_norm);
      }
      if (layout.density_l2_product_self >= 0) {
        NEP_SPIN3_OC_STORE_AT(
            layout.density_l2_product_self + c,
            spin3_oc_dotn<15>(center + base + kSpin3OcQP,
                              center + base + kSpin3OcQP));
      }
      NEP_SPIN3_OC_STORE_AT(
          layout.density_l0_dot_response + c,
          spin3_oc_dotn<3>(m, center + base + kSpin3OcDM));
    }
  }
  if (layout.correlation_same_edge >= 0) {
    const int same_offset = spin3_oc_same_offset(C);
    for (int left = 0; left < C; ++left) {
      for (int right = left; right < C; ++right) {
        const int pair = spin3_oc_pair_index(C, left, right);
        const float same = center[same_offset + pair];
        NEP_SPIN3_OC_STORE_AT(layout.correlation_same_edge + pair, same);
        NEP_SPIN3_OC_STORE_AT(
            layout.correlation_distinct_neighbor + pair,
            first_dot[left] * first_dot[right] - same);
        if (layout.correlation_distinct_l1 >= 0) {
          NEP_SPIN3_OC_STORE_AT(
              layout.correlation_distinct_l1 + pair,
              spin3_oc_dotn<3>(
                  center + layout.angular_l1_moment_offset + 3 * left,
                  center + layout.angular_l1_moment_offset + 3 * right) - same);
        }
        if (layout.correlation_distinct_l2 >= 0) {
          NEP_SPIN3_OC_STORE_AT(
              layout.correlation_distinct_l2 + pair,
              1.5f * spin3_oc_dotn<5>(
                  center + layout.angular_l2_moment_offset + 5 * left,
                  center + layout.angular_l2_moment_offset + 5 * right) - same);
        }
      }
    }
  }
  for (int row = 0; row < C; ++row) {
    if (layout.edge_l11_axial >= 0) {
      float m0[3], p0[3], p1[3], x0[3], l0[1], t0[5], w0[3];
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcM, 0, row, m0);
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcP, 0, row, p0);
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcP, 1, row, p1);
      spin3_oc_project_density<1>(center, projection, C, kSpin3OcL, 0, row, l0);
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcX, 0, row, x0);
      spin3_oc_project_density<5>(center, projection, C, kSpin3OcT, 0, row, t0);
      spin3_cross3(si, m0, w0);
      float p_axis[3], l11_axis[3];
      spin3_cross3(p0, p1, p_axis);
      spin3_oc_l11_axis_from_canonical(si, l0[0], x0, t0, l11_axis);
      if (layout.coupling_l11_axial >= 0) {
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l11_axial + row, spin3_dot3(p_axis, w0));
      }
      NEP_SPIN3_OC_STORE_AT(
          layout.edge_l11_axial + row, spin3_dot3(p0, l11_axis));
      if (layout.coupling_l11_dot_response >= 0) {
        float dm0[3], dw0[3];
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcDM, 0, row, dm0);
        spin3_cross3(si, dm0, dw0);
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l11_dot_response + row,
            spin3_dot3(p_axis, dw0));
      }
      if (layout.coupling_l111_p_m_x >= 0) {
        float m1[3], x2[3], cross_value[3];
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcM, 1, row, m1);
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcX, 2, row, x2);
        spin3_cross3(m1, x2, cross_value);
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l111_p_m_x + row,
            spin3_dot3(p0, cross_value));
      }
      if (layout.coupling_l111_bulk >= 0) {
        float p2[3], x3[3], cross12[3];
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcP, 2, row, p2);
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcX, 3, row, x3);
        spin3_cross3(p1, p2, cross12);
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l111_bulk + row,
            -spin3_dot3(p0, cross12) * spin3_dot3(si, x3));
      }
    }
    if (layout.edge_l22_axial >= 0) {
      float m0[3], p0[3], p1[3], x0[3], q0[5], q1[5], qp0[15];
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcM, 0, row, m0);
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcP, 0, row, p0);
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcP, 1, row, p1);
      spin3_oc_project_density<3>(center, projection, C, kSpin3OcX, 0, row, x0);
      spin3_oc_project_density<5>(center, projection, C, kSpin3OcQ, 0, row, q0);
      spin3_oc_project_density<5>(center, projection, C, kSpin3OcQ, 1, row, q1);
      spin3_oc_project_density<15>(center, projection, C, kSpin3OcQP, 0, row, qp0);
      float q_axis[3], w0[3];
      spin3_oc_axial_commutator(q0, q1, q_axis);
      spin3_cross3(si, m0, w0);
      if (layout.coupling_l22_axial >= 0) {
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l22_axial + row, spin3_dot3(q_axis, w0));
      }
      NEP_SPIN3_OC_STORE_AT(
          layout.edge_l22_axial + row,
          spin3_oc_l22_scalar_from_canonical(si, m0, q0, qp0));
      if (layout.coupling_l22_dot_response >= 0) {
        float dm0[3], dw0[3];
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcDM, 0, row, dm0);
        spin3_cross3(si, dm0, dw0);
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l22_dot_response + row,
            spin3_dot3(q_axis, dw0));
      }
      if (layout.coupling_l111_p_qs_x >= 0) {
        float qp1[15], qs1[3], x2[3], cross_value[3];
        spin3_oc_project_density<15>(center, projection, C, kSpin3OcQP, 1, row, qp1);
        spin3_oc_project_density<3>(center, projection, C, kSpin3OcX, 2, row, x2);
        spin3_oc_qp_vector(qp1, qs1);
        spin3_cross3(qs1, x2, cross_value);
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l111_p_qs_x + row,
            spin3_dot3(p0, cross_value));
      }
      if (layout.coupling_l112_edge_response >= 0) {
        float q_on_p1[3], mixed_axis[3];
        spin3_oc_stf5_matvec(q0, p1, q_on_p1);
        spin3_cross3(p1, q_on_p1, mixed_axis);
        NEP_SPIN3_OC_STORE_AT(
            layout.coupling_l112_edge_response + row,
            -spin3_dot3(p0, mixed_axis) * spin3_dot3(si, x0));
      }
    }
  }
#undef NEP_SPIN3_OC_STORE_AT
}
