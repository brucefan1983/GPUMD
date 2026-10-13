/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    Copyright 2026 NEPAdapters contributors
    This file is part of GPUMD and is distributed under GPLv3 or later.

    The Spin3 protocol follows TorchNEP spin commit 3e2a5b3 and the matching
    NEPAdapters dev implementation inspected on 2026-09-03 (GPLv3+).
*/

#include "nep_spin.cuh"
#include "nep_kernels.cuh"
#include "nep_small_box.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

using SimulationBox = Box;

#include "nep_spin3_common.cuh"

enum class SpinVirialMode : int {
  disabled,
  center_owned,
  neighbor_owned,
  center_and_neighbor_float_sink,
};

constexpr const char* kElementSymbols[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

__device__ __forceinline__ void atomic_add_per_atom_virial_double(
  const int atom_stride,
  const int atom,
  const float x12,
  const float y12,
  const float z12,
  const float fx,
  const float fy,
  const float fz,
  double* virial)
{
  atomicAdd(&virial[atom], -static_cast<double>(x12 * fx));
  atomicAdd(&virial[atom_stride + atom], -static_cast<double>(y12 * fy));
  atomicAdd(&virial[2 * atom_stride + atom], -static_cast<double>(z12 * fz));
  atomicAdd(&virial[3 * atom_stride + atom], -static_cast<double>(x12 * fy));
  atomicAdd(&virial[4 * atom_stride + atom], -static_cast<double>(x12 * fz));
  atomicAdd(&virial[5 * atom_stride + atom], -static_cast<double>(y12 * fz));
  atomicAdd(&virial[6 * atom_stride + atom], -static_cast<double>(y12 * fx));
  atomicAdd(&virial[7 * atom_stride + atom], -static_cast<double>(z12 * fx));
  atomicAdd(&virial[8 * atom_stride + atom], -static_cast<double>(z12 * fy));
}

__device__ __forceinline__ void atomic_add_per_atom_virial_float(
  const int atom_stride,
  const int atom,
  const float x12,
  const float y12,
  const float z12,
  const float fx,
  const float fy,
  const float fz,
  float* virial)
{
  atomicAdd(&virial[atom], -x12 * fx);
  atomicAdd(&virial[atom_stride + atom], -y12 * fy);
  atomicAdd(&virial[2 * atom_stride + atom], -z12 * fz);
  atomicAdd(&virial[3 * atom_stride + atom], -x12 * fy);
  atomicAdd(&virial[4 * atom_stride + atom], -x12 * fz);
  atomicAdd(&virial[5 * atom_stride + atom], -y12 * fz);
  atomicAdd(&virial[6 * atom_stride + atom], -y12 * fx);
  atomicAdd(&virial[7 * atom_stride + atom], -z12 * fx);
  atomicAdd(&virial[8 * atom_stride + atom], -z12 * fy);
}

__device__ __forceinline__ void atomic_add_spin_transfer_float(
  int, int, float, float, float, const float*, float*)
{
}

__global__ void mask_inactive_spin_mforce(
  const int atom_count,
  const int atom_stride,
  const int* __restrict__ types,
  const int* __restrict__ spin_dof_type_active,
  double* __restrict__ mforce)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < atom_count && spin_dof_type_active[types[atom]] == 0) {
    mforce[atom] = 0.0;
    mforce[atom_stride + atom] = 0.0;
    mforce[2 * atom_stride + atom] = 0.0;
  }
}

using SpinPolynomialLayout = NEP_Spin::Spin_Polynomial_Layout;

SpinPolynomialLayout make_spin_polynomial_layout(
  const int channels, const int l_max, const int order, const int soc)
{
  SpinPolynomialLayout layout;
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
  layout.edge_l0_dot = offset; offset += channels;
  layout.edge_l0_neighbor_s2 = offset; offset += channels;
  if (soc != 0 && l_max >= 2) {
    layout.edge_l2_pair = offset; offset += channels;
    layout.center_l2_environment = offset; offset += channels;
  }
  if (order >= 2) {
    layout.edge_l0_dot2 = offset; offset += channels;
    layout.density_l0_self = offset; offset += channels;
    if (l_max >= 1) {
      if (soc != 0) {
        layout.density_l1_longitudinal_self = offset; offset += channels;
        layout.density_l1_axial_self = offset; offset += channels;
        layout.density_l1_stf_self = offset; offset += channels;
      } else {
        layout.density_l1_product_self = offset; offset += channels;
      }
    }
    if (l_max >= 2) {
      layout.density_l2_product_self = offset; offset += channels;
    }
    layout.density_l0_dot_response = offset; offset += channels;
    layout.correlation_same_edge = offset; offset += layout.pair_count;
    layout.correlation_distinct_neighbor = offset; offset += layout.pair_count;
    if (l_max >= 1) {
      layout.correlation_distinct_l1 = offset; offset += layout.pair_count;
    }
    if (l_max >= 2) {
      layout.correlation_distinct_l2 = offset; offset += layout.pair_count;
    }
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) {
        layout.coupling_l11_axial = offset; offset += channels;
      }
      layout.edge_l11_axial = offset; offset += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) {
        layout.coupling_l22_axial = offset; offset += channels;
      }
      layout.edge_l22_axial = offset; offset += channels;
    }
  }
  if (order >= 3) {
    layout.edge_l0_moment_gate = offset; offset += channels;
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) {
        layout.coupling_l11_dot_response = offset; offset += channels;
      }
      layout.coupling_l111_p_m_x = offset; offset += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) {
        layout.coupling_l22_dot_response = offset; offset += channels;
      }
      layout.coupling_l111_p_qs_x = offset; offset += channels;
      if (channels >= 2) {
        layout.coupling_l112_edge_response = offset; offset += channels;
      }
    }
    if (soc != 0 && l_max >= 1 && channels >= 3) {
      layout.coupling_l111_bulk = offset; offset += channels;
    }
  }
  layout.descriptor_dim = offset;
  return layout;
}

template <int C, bool NeedDerivatives>
__device__ __forceinline__ void evaluate_spin3_edge_weights_b9_f32(
  const float spin_cutoff,
  const float dist,
  const int num_types,
  const int type_pair,
  const float* __restrict__ descriptor_coefficients,
  const int spin_coefficient_offset,
  float* weights,
  float* weight_derivatives)
{
  constexpr int BasisCount = 9;
  constexpr float Pi = 3.14159265358979323846f;
  const int pair_count = num_types * num_types;
  const float rcinv = 1.0f / spin_cutoff;
  const float r_scaled = dist * rcinv;
  const float phase = Pi * r_scaled;
  const float fc = 0.5f * cosf(phase) + 0.5f;
  const float shifted = r_scaled - 1.0f;
  const float x = 2.0f * shifted * shifted - 1.0f;
  const float dxdr = 2.0f * shifted * rcinv;
  const float fcp = NeedDerivatives ? -0.5f * Pi * sinf(phase) * rcinv : 0.0f;
  float fn[BasisCount] = {};
  float fnp[BasisCount] = {};
  fn[0] = fc;
  fnp[0] = fcp;
  const float raw1 = 0.5f * (x + 1.0f);
  fn[1] = raw1 * fc;
  fnp[1] = dxdr * fc + raw1 * fcp;
  float t0 = 1.0f;
  float t1 = x;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  for (int n = 2; n < BasisCount; ++n) {
    const float t2 = 2.0f * x * t1 - t0;
    const float raw = 0.5f * (t2 + 1.0f);
    fn[n] = raw * fc;
    fnp[n] = static_cast<float>(n) * u1 * dxdr * fc + raw * fcp;
    const float u2 = 2.0f * x * u1 - u0;
    t0 = t1; t1 = t2; u0 = u1; u1 = u2;
  }
  for (int c = 0; c < C; ++c) {
    float weight = 0.0f;
    float derivative = 0.0f;
    for (int k = 0; k < BasisCount; ++k) {
      const int index = spin_coefficient_offset +
        (c * BasisCount + k) * pair_count + type_pair;
      weight += fn[k] * descriptor_coefficients[index];
      if constexpr (NeedDerivatives) {
        derivative += fnp[k] * descriptor_coefficients[index];
      }
    }
    weights[c] = weight;
    if constexpr (NeedDerivatives) {
      weight_derivatives[c] = derivative;
    }
  }
}

template <int C, bool NeedDerivatives = true>
__device__ __forceinline__ bool load_spin3_edge_f32(
  const int atom,
  const int neighbor,
  const int atom_stride,
  const int num_types,
  const int spin_basis_size,
  const float spin_cutoff,
  const float* __restrict__ spin_cutoff_pair,
  const SimulationBox box,
  const int* __restrict__ types,
  const double* __restrict__ positions_soa3,
  const double* __restrict__ slot_r12,
  const int r12_plane_size,
  const int slot_index,
  const double* __restrict__ spins_soa3,
  const float* __restrict__ descriptor_coefficients,
  const int spin_coefficient_offset,
  float* rhat,
  float& dist,
  float* si,
  float* sj,
  float* weights,
  float* weight_derivatives)
{
  if (slot_r12) {
    const float dx = static_cast<float>(slot_r12[slot_index]);
    const float dy = static_cast<float>(slot_r12[r12_plane_size + slot_index]);
    const float dz = static_cast<float>(slot_r12[2 * r12_plane_size + slot_index]);
    dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist > 1.0e-12f) {
      const float inverse_distance = 1.0f / dist;
      rhat[0] = dx * inverse_distance;
      rhat[1] = dy * inverse_distance;
      rhat[2] = dz * inverse_distance;
    }
  } else {
    compute_spin_edge_geometry_f32(
      atom, neighbor, atom_stride, box, positions_soa3, rhat, dist);
  }
  for (int d = 0; d < 3; ++d) {
    si[d] = static_cast<float>(spins_soa3[d * atom_stride + atom]);
    sj[d] = static_cast<float>(spins_soa3[d * atom_stride + neighbor]);
  }
  const int type_pair = types[atom] * num_types + types[neighbor];
  const float resolved_spin_cutoff =
    spin_cutoff_pair == nullptr ? spin_cutoff : spin_cutoff_pair[type_pair];
  if (!(dist > 1.0e-12f && dist < resolved_spin_cutoff) || spin_basis_size != 8) {
    return false;
  }
  evaluate_spin3_edge_weights_b9_f32<C, NeedDerivatives>(
    resolved_spin_cutoff,
    dist,
    num_types,
    type_pair,
    descriptor_coefficients,
    spin_coefficient_offset,
    weights,
    weight_derivatives);
  return true;
}

#include "nep_spin3_layout.cuh"
#include "nep_spin3_descriptor.cuh"
#include "nep_spin3_force.cuh"

__device__ int required_small_box_capacity[1];

__global__ void find_local_neighbors_spin(
  const int N,
  const Box box,
  const float cutoff_radial_square,
  const float cutoff_angular_square,
  const float cutoff_spin_square,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const int* __restrict__ NN_global,
  const int* __restrict__ NL_global,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  int* NN_spin,
  int* NL_spin)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }

  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int count_radial = 0;
  int count_angular = 0;
  int count_spin = 0;

  for (int slot = 0; slot < NN_global[n1]; ++slot) {
    const int n2 = NL_global[static_cast<std::size_t>(N) * slot + n1];
    float x12 = x[n2] - x1;
    float y12 = y[n2] - y1;
    float z12 = z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    const float distance_square = x12 * x12 + y12 * y12 + z12 * z12;

    if (distance_square < cutoff_radial_square) {
      NL_radial[static_cast<std::size_t>(N) * count_radial++ + n1] = n2;
    }
    if (distance_square < cutoff_angular_square) {
      NL_angular[static_cast<std::size_t>(N) * count_angular++ + n1] = n2;
    }
    if (distance_square < cutoff_spin_square) {
      NL_spin[static_cast<std::size_t>(N) * count_spin++ + n1] = n2;
    }
  }

  NN_radial[n1] = count_radial;
  NN_angular[n1] = count_angular;
  NN_spin[n1] = count_spin;
}

__device__ __forceinline__ void apply_mic_spin_small_box(
  const Box& box, const NEP::ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
  double sx12 =
    (box.cpu_h[9] * x12 + box.cpu_h[10] * y12 + box.cpu_h[11] * z12) /
    ebox.num_cells[0];
  double sy12 =
    (box.cpu_h[12] * x12 + box.cpu_h[13] * y12 + box.cpu_h[14] * z12) /
    ebox.num_cells[1];
  double sz12 =
    (box.cpu_h[15] * x12 + box.cpu_h[16] * y12 + box.cpu_h[17] * z12) /
    ebox.num_cells[2];
  if (box.pbc_x == 1)
    sx12 -= nearbyint(sx12);
  if (box.pbc_y == 1)
    sy12 -= nearbyint(sy12);
  if (box.pbc_z == 1)
    sz12 -= nearbyint(sz12);
  x12 = box.cpu_h[0] * ebox.num_cells[0] * sx12 +
        box.cpu_h[1] * ebox.num_cells[1] * sy12 +
        box.cpu_h[2] * ebox.num_cells[2] * sz12;
  y12 = box.cpu_h[3] * ebox.num_cells[0] * sx12 +
        box.cpu_h[4] * ebox.num_cells[1] * sy12 +
        box.cpu_h[5] * ebox.num_cells[2] * sz12;
  z12 = box.cpu_h[6] * ebox.num_cells[0] * sx12 +
        box.cpu_h[7] * ebox.num_cells[1] * sy12 +
        box.cpu_h[8] * ebox.num_cells[2] * sz12;
}

__global__ void find_neighbor_list_spin_small_box(
  const NEP::ParaMB paramb,
  const float spin_cutoff,
  const int N,
  const Box box,
  const NEP::ExpandedBox ebox,
  const int capacity,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  int* NN_spin,
  int* NL_spin,
  double* r12_radial,
  double* r12_angular,
  double* r12_spin,
  int* required_capacity)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  const int t1 = type[n1];
  const int plane_size = N * capacity;
  int count_radial = 0;
  int count_angular = 0;
  int count_spin = 0;
  for (int n2 = 0; n2 < N; ++n2) {
    for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
      for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
        for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
          if (n1 == n2 && ia == 0 && ib == 0 && ic == 0) {
            continue;
          }
          double x12 = x[n2] + box.cpu_h[0] * ia + box.cpu_h[1] * ib +
                       box.cpu_h[2] * ic - x1;
          double y12 = y[n2] + box.cpu_h[3] * ia + box.cpu_h[4] * ib +
                       box.cpu_h[5] * ic - y1;
          double z12 = z[n2] + box.cpu_h[6] * ia + box.cpu_h[7] * ib +
                       box.cpu_h[8] * ic - z1;
          apply_mic_spin_small_box(box, ebox, x12, y12, z12);
          const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
          const int t2 = type[n2];
          const float rc_radial = 0.5f * (paramb.rc_radial[t1] + paramb.rc_radial[t2]);
          const float rc_angular = 0.5f * (paramb.rc_angular[t1] + paramb.rc_angular[t2]);
          if (d2 < rc_radial * rc_radial) {
            const int slot = count_radial++;
            if (slot < capacity) {
              const int index = n1 + N * slot;
              NL_radial[index] = n2;
              r12_radial[index] = x12;
              r12_radial[plane_size + index] = y12;
              r12_radial[2 * plane_size + index] = z12;
            }
          }
          if (d2 < rc_angular * rc_angular) {
            const int slot = count_angular++;
            if (slot < capacity) {
              const int index = n1 + N * slot;
              NL_angular[index] = n2;
              r12_angular[index] = x12;
              r12_angular[plane_size + index] = y12;
              r12_angular[2 * plane_size + index] = z12;
            }
          }
          if (d2 < spin_cutoff * spin_cutoff) {
            const int slot = count_spin++;
            if (slot < capacity) {
              const int index = n1 + N * slot;
              NL_spin[index] = n2;
              r12_spin[index] = x12;
              r12_spin[plane_size + index] = y12;
              r12_spin[2 * plane_size + index] = z12;
            }
          }
        }
      }
    }
  }
  NN_radial[n1] = min(count_radial, capacity);
  NN_angular[n1] = min(count_angular, capacity);
  NN_spin[n1] = min(count_spin, capacity);
  atomicMax(required_capacity, max(count_radial, max(count_angular, count_spin)));
}

__global__ void find_structural_descriptor(
  const NEP::ParaMB paramb,
  const NEP::ANN ann,
  const int N,
  const Box box,
  const int* NN_radial,
  const int* NL_radial,
  const int* NN_angular,
  const int* NL_angular,
  const double* r12_radial,
  const double* r12_angular,
  const int r12_plane_size,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  float* descriptor,
  float* sum_fxyz)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) {
    return;
  }
  const int t1 = type[n1];
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  float q[MAX_DIM] = {0.0f};

  for (int i1 = 0; i1 < NN_radial[n1]; ++i1) {
    const int n2 = NL_radial[static_cast<size_t>(N) * i1 + n1];
    const int slot = N * i1 + n1;
    double x12 = r12_radial ? r12_radial[slot] : x[n2] - x1;
    double y12 = r12_radial ? r12_radial[r12_plane_size + slot] : y[n2] - y1;
    double z12 = r12_radial ? r12_radial[2 * r12_plane_size + slot] : z[n2] - z1;
    if (!r12_radial) {
      apply_mic(box, x12, y12, z12);
    }
    const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    const int t2 = type[n2];
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
    const int basis_count =
      (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        const int basis = n * (paramb.basis_size_radial + 1) + k;
        gn12 += fn12[k] * ann.c_type_pair[(t1 * paramb.num_types + t2) * basis_count + basis];
      }
      q[n] += gn12;
    }
  }

  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    float s[NUM_OF_ABC] = {0.0f};
    const int abc_count = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
    for (int i1 = 0; i1 < NN_angular[n1]; ++i1) {
      const int n2 = NL_angular[n1 + N * i1];
      const int slot = N * i1 + n1;
      double x12 = r12_angular ? r12_angular[slot] : x[n2] - x1;
      double y12 = r12_angular ? r12_angular[r12_plane_size + slot] : y[n2] - y1;
      double z12 = r12_angular ? r12_angular[2 * r12_plane_size + slot] : z[n2] - z1;
      if (!r12_angular) {
        apply_mic(box, x12, y12, z12);
      }
      const float d12 = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
      const int t2 = type[n2];
      const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
      float gn12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = paramb.num_c_radial;
        c_index += (t1 * paramb.num_types + t2) *
                   ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
        c_index += n * (paramb.basis_size_angular + 1) + k;
        gn12 += fn12[k] * ann.c_type_pair[c_index];
      }
      accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
    }
    find_q(
      paramb.L_max,
      paramb.has_q_222,
      paramb.has_q_1111,
      paramb.has_q_112,
      paramb.has_q_123,
      paramb.has_q_233,
      paramb.has_q_134,
      paramb.n_max_angular + 1,
      n,
      s,
      q + paramb.n_max_radial + 1);
    for (int abc = 0; abc < abc_count; ++abc) {
      sum_fxyz[static_cast<size_t>(N) * (n * abc_count + abc) + n1] = s[abc];
    }
  }
  const int struct_dim = paramb.n_max_radial + 1 + paramb.dim_angular;
  for (int d = 0; d < struct_dim; ++d) {
    descriptor[static_cast<size_t>(N) * d + n1] = q[d];
  }
}

__global__ void find_potential(
  const int N,
  const int descriptor_dim,
  const int hidden_neurons,
  const int num_types,
  const int* type,
  const float* descriptor,
  const float* parameters,
  const float* q_scaler,
  const float* spin_baseline,
  double* potential,
  float* Fp)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= N) {
    return;
  }
  const int atom_type = type[atom];
  const std::size_t type_block =
    static_cast<std::size_t>(hidden_neurons) * (descriptor_dim + 2);
  const float* w0 = parameters + static_cast<std::size_t>(atom_type) * type_block;
  const float* b0 = w0 + static_cast<std::size_t>(hidden_neurons) * descriptor_dim;
  const float* w1 = b0 + hidden_neurons;
  const float* b1 = parameters + static_cast<std::size_t>(num_types) * type_block;
  float q[MAX_DIM_SPIN] = {0.0f};
  for (int d = 0; d < descriptor_dim; ++d) {
    q[d] = descriptor[static_cast<std::size_t>(N) * d + atom] * q_scaler[d];
  }
  float energy = 0.0f;
  float fp[MAX_DIM_SPIN] = {0.0f};
  for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
    float projection = 0.0f;
    for (int d = 0; d < descriptor_dim; ++d) {
      projection += w0[neuron * descriptor_dim + d] * q[d];
    }
    const float activation = tanhf(projection - b0[neuron]);
    const float activation_derivative = 1.0f - activation * activation;
    energy += w1[neuron] * activation;
    for (int d = 0; d < descriptor_dim; ++d) {
      fp[d] += w1[neuron] * activation_derivative *
               w0[neuron * descriptor_dim + d];
    }
  }
  energy -= b1[0];
  potential[atom] +=
    static_cast<double>(energy) + static_cast<double>(spin_baseline[atom_type]);
  for (int d = 0; d < descriptor_dim; ++d) {
    Fp[static_cast<std::size_t>(N) * d + atom] = fp[d] * q_scaler[d];
  }
}

template <int lanes_per_atom>
__global__ void find_potential_warp(
  const int N,
  const int descriptor_dim,
  const int hidden_neurons,
  const int num_types,
  const int* __restrict__ type,
  const float* __restrict__ descriptor,
  const float* __restrict__ parameters,
  const float* __restrict__ q_scaler,
  const float* __restrict__ spin_baseline,
  double* potential,
  float* __restrict__ Fp)
{
  static_assert(32 % lanes_per_atom == 0, "one warp must contain complete atom tiles");
  constexpr int warps_per_block = 4;
  constexpr int atoms_per_warp = 32 / lanes_per_atom;
  constexpr int atoms_per_block = warps_per_block * atoms_per_warp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int atom_lane = lane / lanes_per_atom;
  const int work_lane = lane - atom_lane * lanes_per_atom;
  const int atom_local = warp * atoms_per_warp + atom_lane;
  const int atom = blockIdx.x * atoms_per_block + atom_local;
  const unsigned int atom_mask =
    ((1u << lanes_per_atom) - 1u) << (atom_lane * lanes_per_atom);

  extern __shared__ float state[];
  float* q = state + static_cast<std::size_t>(atom_local) * descriptor_dim;
  float* hidden_delta =
    state + static_cast<std::size_t>(atoms_per_block) * descriptor_dim +
    static_cast<std::size_t>(atom_local) * hidden_neurons;

  if (atom < N) {
    for (int d = work_lane; d < descriptor_dim; d += lanes_per_atom) {
      q[d] = descriptor[static_cast<std::size_t>(N) * d + atom] * q_scaler[d];
    }
  }
  __syncwarp(atom_mask);
  if (atom >= N) {
    return;
  }

  const int atom_type = type[atom];
  const std::size_t type_block =
    static_cast<std::size_t>(hidden_neurons) * (descriptor_dim + 2);
  const float* w0 = parameters + static_cast<std::size_t>(atom_type) * type_block;
  const float* b0 = w0 + static_cast<std::size_t>(hidden_neurons) * descriptor_dim;
  const float* w1 = b0 + hidden_neurons;
  const float* b1 = parameters + static_cast<std::size_t>(num_types) * type_block;

  float energy = 0.0f;
  for (int neuron = work_lane; neuron < hidden_neurons; neuron += lanes_per_atom) {
    float projection = 0.0f;
    for (int d = 0; d < descriptor_dim; ++d) {
      projection += w0[neuron * descriptor_dim + d] * q[d];
    }
    const float activation = tanhf(projection - b0[neuron]);
    const float delta = w1[neuron] * (1.0f - activation * activation);
    hidden_delta[neuron] = delta;
    energy += w1[neuron] * activation;
  }
  for (int offset = lanes_per_atom / 2; offset > 0; offset >>= 1) {
    energy += __shfl_down_sync(atom_mask, energy, offset, lanes_per_atom);
  }
  __syncwarp(atom_mask);

  if (work_lane == 0) {
    energy -= b1[0];
    potential[atom] +=
      static_cast<double>(energy) + static_cast<double>(spin_baseline[atom_type]);
  }
  for (int d = work_lane; d < descriptor_dim; d += lanes_per_atom) {
    float fp = 0.0f;
    for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
      fp += hidden_delta[neuron] * w0[neuron * descriptor_dim + d];
    }
    Fp[static_cast<std::size_t>(N) * d + atom] = fp * q_scaler[d];
  }
}

template <int atoms_per_warp, int edge_lanes, bool l4_q222_only>
__global__ void find_force_angular_warp(
  const NEP::ParaMB paramb,
  const NEP::ANN ann,
  const int N,
  const Box box,
  const int* NN_angular,
  const int* NL_angular,
  const int* __restrict__ type,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const float* __restrict__ Fp,
  const float* __restrict__ sum_fxyz,
  double* force,
  double* virial)
{
  static_assert(atoms_per_warp * edge_lanes == 32, "one tile must fill one warp");
  constexpr int warps_per_block = 4;
  constexpr int centers_per_block = warps_per_block * atoms_per_warp;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int atom_lane = lane / edge_lanes;
  const int edge_lane = lane - atom_lane * edge_lanes;
  unsigned int center_mask;
  if constexpr (edge_lanes == 32) {
    center_mask = 0xffffffffu;
  } else {
    center_mask =
      ((1u << edge_lanes) - 1u) << (atom_lane * edge_lanes);
  }
  const int center_local = warp * atoms_per_warp + atom_lane;
  const int n1 = blockIdx.x * centers_per_block + center_local;

  const int abc_count = (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const int sum_count = (paramb.n_max_angular + 1) * NUM_OF_ABC;
  const int state_size = paramb.dim_angular + sum_count;
  extern __shared__ float center_state[];
  float* center_Fp = center_state + center_local * state_size;
  float* center_sum_fxyz = center_Fp + paramb.dim_angular;

  if (n1 < N) {
    for (int d = edge_lane; d < paramb.dim_angular; d += edge_lanes) {
      center_Fp[d] = Fp[static_cast<size_t>(N) * (paramb.n_max_radial + 1 + d) + n1];
    }
    const int compact_sum_count = (paramb.n_max_angular + 1) * abc_count;
    for (int d = edge_lane; d < compact_sum_count; d += edge_lanes) {
      const int n = d / abc_count;
      const int abc = d - n * abc_count;
      center_sum_fxyz[n * NUM_OF_ABC + abc] =
        sum_fxyz[static_cast<size_t>(N) * d + n1];
    }
  }
  __syncwarp();

  if (n1 >= N) {
    return;
  }
  const int t1 = type[n1];
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  float center_force[3] = {};
  for (int i1 = edge_lane; i1 < NN_angular[n1]; i1 += edge_lanes) {
    const int index = i1 * N + n1;
    const int n2 = NL_angular[index];
    float x12 = x[n2] - x1;
    float y12 = y[n2] - y1;
    float z12 = z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float r12[3] = {x12, y12, z12};
    const float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float f12[3] = {0.0f};
    float fc12;
    float fcp12;
    const int t2 = type[n2];
    const float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = paramb.num_c_radial;
        c_index += (t1 * paramb.num_types + t2) *
                   ((paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
        c_index += n * (paramb.basis_size_angular + 1) + k;
        gn12 += fn12[k] * ann.c_type_pair[c_index];
        gnp12 += fnp12[k] * ann.c_type_pair[c_index];
      }
      if (l4_q222_only) {
        accumulate_f12(
          4,
          true,
          false,
          false,
          false,
          false,
          false,
          5,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          center_Fp,
          center_sum_fxyz,
          f12);
      } else {
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222,
          paramb.has_q_1111,
          paramb.has_q_112,
          paramb.has_q_123,
          paramb.has_q_233,
          paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          center_Fp,
          center_sum_fxyz,
          f12);
      }
    }
    // One directed response contributes +f12 to its center and -f12 to its
    // neighbor. The neighbor-owned virial is (-r12) tensor f12.
    for (int d = 0; d < 3; ++d) {
      center_force[d] += f12[d];
      atomicAdd(force + d * N + n2, -static_cast<double>(f12[d]));
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const int component = virial_internal_component(3 * a + b);
        atomicAdd(
          virial + component * N + n2,
          -static_cast<double>(r12[a] * f12[b]));
      }
    }
  }
  for (int offset = edge_lanes / 2; offset > 0; offset >>= 1) {
    for (int d = 0; d < 3; ++d) {
      center_force[d] +=
        __shfl_down_sync(center_mask, center_force[d], offset, edge_lanes);
    }
  }
  if (edge_lane == 0) {
    for (int d = 0; d < 3; ++d) {
      atomicAdd(force + d * N + n1, static_cast<double>(center_force[d]));
    }
  }
}

template <int atoms_per_warp, int edge_lanes, bool l4_q222_only>
void launch_force_angular_warp(
  const NEP::ParaMB& paramb,
  const NEP::ANN& ann,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  const int grid_size,
  const int block_size,
  const std::size_t shared_size)
{
  const int N = type.size();
  find_force_angular_warp<atoms_per_warp, edge_lanes, l4_q222_only>
    <<<grid_size, block_size, shared_size>>>(
    paramb,
    ann,
    N,
    box,
    data.NN_angular.data(),
    data.NL_angular.data(),
    type.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    data.Fp.data(),
    data.sum_fxyz.data(),
    force.data(),
    virial.data());
}

std::vector<std::string> split(const std::string& line)
{
  std::istringstream input(line);
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string> next_tokens(std::ifstream& input)
{
  std::string line;
  while (std::getline(input, line)) {
    std::vector<std::string> tokens = split(line);
    if (!tokens.empty()) {
      return tokens;
    }
  }
  return {};
}

int parse_int(const std::string& token)
{
  std::size_t end = 0;
  const long value = std::stol(token, &end);
  if (end != token.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid integer token: " + token);
  }
  return static_cast<int>(value);
}

double parse_double(const std::string& token)
{
  std::size_t end = 0;
  const double value = std::stod(token, &end);
  if (end != token.size() || !std::isfinite(value)) {
    throw std::runtime_error("invalid finite numeric token: " + token);
  }
  return value;
}

float parse_float(const std::string& token)
{
  const double value = parse_double(token);
  if (std::abs(value) > std::numeric_limits<float>::max()) {
    throw std::runtime_error("model parameter is outside the float range");
  }
  return static_cast<float>(value);
}

void require_line(
  const std::vector<std::string>& tokens, const char* name, const std::size_t arity)
{
  if (tokens.size() != arity || tokens.empty() || tokens[0] != name) {
    throw std::runtime_error(
      std::string("expected exact '") + name + "' line with " +
      std::to_string(arity - 1) + " value(s)");
  }
}

bool parse_flag(const std::string& token, const char* name, const bool allow_legacy_two = false)
{
  const int value = parse_int(token);
  if (value != 0 && value != 1 && !(allow_legacy_two && value == 2)) {
    throw std::runtime_error(std::string(name) + " must be 0 or 1");
  }
  return value != 0;
}

void check_unique_elements(const std::vector<std::string>& elements)
{
  const std::set<std::string> unique(elements.begin(), elements.end());
  if (unique.size() != elements.size()) {
    throw std::runtime_error("model element symbols must be unique");
  }
}

std::vector<int> parse_active_types(
  const std::vector<std::string>& tokens,
  const std::vector<std::string>& elements,
  const char* name)
{
  if (tokens.size() < 2) {
    throw std::runtime_error(std::string(name) + " must enable at least one model type");
  }
  std::vector<int> active(elements.size(), 0);
  for (std::size_t index = 1; index < tokens.size(); ++index) {
    const auto found = std::find(elements.begin(), elements.end(), tokens[index]);
    if (found == elements.end()) {
      throw std::runtime_error(std::string("unknown ") + name + " model type");
    }
    const std::size_t type = static_cast<std::size_t>(found - elements.begin());
    if (active[type] != 0) {
      throw std::runtime_error(std::string("duplicate model type in ") + name);
    }
    active[type] = 1;
  }
  return active;
}

template <int C>
void launch_spin3_descriptors_typed(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0,
  gpuStream_t stream = nullptr)
{
  constexpr int block_size = 128;
  const int N = type.size();
  const int density_work = C * N;
  build_spin3_oc_density_bank<C><<<
    (density_work + block_size - 1) / block_size, block_size, 0, stream>>>(
    model.spin_polynomial_layout,
    N,
    N,
    model.struct_descriptor_dim,
    model.num_types,
    model.spin_basis_size[0],
    static_cast<float>(model.spin_cutoff),
    data.spin_cutoff_pair.data(),
    static_cast<int>(model.radial_parameter_count + model.angular_parameter_count),
    box,
    type.data(),
    data.spin_dof_type_active.data(),
    data.spin_env_type_active.data(),
    position.data(),
    slot_r12,
    r12_plane_size,
    spin.data(),
    data.NN_spin.data(),
    data.NL_spin.data(),
    data.descriptor_parameters_type_pair.data(),
    data.descriptor.data(),
    data.spin3_moments.data());
  contract_spin3_oc_descriptors<C><<<
    (N + block_size - 1) / block_size, block_size, 0, stream>>>(
    model.spin_polynomial_layout,
    N,
    N,
    model.struct_descriptor_dim,
    type.data(),
    data.spin_dof_type_active.data(),
    spin.data(),
    data.spin_projection_parameters.data(),
    data.spin3_moments.data(),
    data.descriptor.data());
  GPU_CHECK_KERNEL
}

void launch_spin3_descriptors(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0,
  gpuStream_t stream = nullptr)
{
  switch (model.spin_compress) {
    case 1: launch_spin3_descriptors_typed<1>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 2: launch_spin3_descriptors_typed<2>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 3: launch_spin3_descriptors_typed<3>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 4: launch_spin3_descriptors_typed<4>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 5: launch_spin3_descriptors_typed<5>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 6: launch_spin3_descriptors_typed<6>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 7: launch_spin3_descriptors_typed<7>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 8: launch_spin3_descriptors_typed<8>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
    case 9: launch_spin3_descriptors_typed<9>(model, box, type, position, spin, data, slot_r12, r12_plane_size, stream); break;
  }
}

template <int C>
void launch_spin3_forces_typed(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& mforce,
  GPU_Vector<double>& virial,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  using Tile = Spin3ForceTile<C>;
  constexpr int block_size = Tile::block_size;
  const int N = type.size();
  constexpr int atoms_per_warp = Tile::atoms_per_warp;
  constexpr int centers_per_block = Tile::centers_per_block;
  const int grid_size = (N - 1) / centers_per_block + 1;
  const std::size_t shared_size =
    static_cast<std::size_t>(centers_per_block) *
    model.spin_polynomial_layout.moment_count * sizeof(float);
  accumulate_spin3_oc_native_forces<
    C, SpinVirialMode::center_owned, false, atoms_per_warp>
    <<<grid_size, block_size, shared_size>>>(
      model.spin_polynomial_layout,
      N,
      N,
      model.struct_descriptor_dim,
      model.num_types,
      model.spin_basis_size[0],
      static_cast<float>(model.spin_cutoff),
      data.spin_cutoff_pair.data(),
      box,
      type.data(),
      data.spin_dof_type_active.data(),
      data.spin_env_type_active.data(),
      position.data(),
      slot_r12,
      r12_plane_size,
      spin.data(),
      data.NN_spin.data(),
      data.NL_spin.data(),
      data.Fp.data(),
      data.descriptor_parameters_type_pair.data(),
      data.spin_projection_parameters.data(),
      data.spin3_moments.data(),
      static_cast<int>(model.radial_parameter_count + model.angular_parameter_count),
      force.data(),
      mforce.data(),
      virial.data(),
      nullptr,
      nullptr);
  GPU_CHECK_KERNEL
}

void launch_spin3_forces(
  const NEP_Spin::Model& model,
  const Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  NEP_Spin_Data& data,
  GPU_Vector<double>& force,
  GPU_Vector<double>& mforce,
  GPU_Vector<double>& virial,
  const double* slot_r12 = nullptr,
  int r12_plane_size = 0)
{
  switch (model.spin_compress) {
    case 1: launch_spin3_forces_typed<1>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 2: launch_spin3_forces_typed<2>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 3: launch_spin3_forces_typed<3>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 4: launch_spin3_forces_typed<4>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 5: launch_spin3_forces_typed<5>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 6: launch_spin3_forces_typed<6>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 7: launch_spin3_forces_typed<7>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 8: launch_spin3_forces_typed<8>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
    case 9: launch_spin3_forces_typed<9>(model, box, type, position, spin, data, force, mforce, virial, slot_r12, r12_plane_size); break;
  }
  const bool has_inactive_dof = std::any_of(
    model.spin_dof_type_active.begin(),
    model.spin_dof_type_active.end(),
    [](const int active) { return active == 0; });
  if (has_inactive_dof) {
    constexpr int block_size = 128;
    const int grid_size = (type.size() - 1) / block_size + 1;
    mask_inactive_spin_mforce<<<grid_size, block_size>>>(
      type.size(), type.size(), type.data(), data.spin_dof_type_active.data(), mforce.data());
  }
  GPU_CHECK_KERNEL
}

template <bool SmallBox>
__global__ void find_spin_zbl_force(
  const NEP::ParaMB paramb,
  const int atom_count,
  const NEP::ZBL zbl,
  const Box box,
  const int* neighbor_count,
  const int* neighbor_list,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  const double* r12x,
  const double* r12y,
  const double* r12z,
  double* force_x,
  double* force_y,
  double* force_z,
  double* virial,
  double* potential)
{
  const int atom1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom1 >= atom_count) {
    return;
  }
  float energy = 0.0f;
  float force_sum[3] = {};
  float virial_sum[9] = {};
  const int type1 = type[atom1];
  const int atomic_number1 = zbl.atomic_numbers[type1];
  const float atomic_number_power1 = powf(static_cast<float>(atomic_number1), 0.23f);
  for (int neighbor = 0; neighbor < neighbor_count[atom1]; ++neighbor) {
    const int index = atom1 + atom_count * neighbor;
    const int atom2 = neighbor_list[index];
    float displacement[3];
    if constexpr (SmallBox) {
      displacement[0] = static_cast<float>(r12x[index]);
      displacement[1] = static_cast<float>(r12y[index]);
      displacement[2] = static_cast<float>(r12z[index]);
    } else {
      displacement[0] = static_cast<float>(x[atom2] - x[atom1]);
      displacement[1] = static_cast<float>(y[atom2] - y[atom1]);
      displacement[2] = static_cast<float>(z[atom2] - z[atom1]);
      apply_mic(box, displacement[0], displacement[1], displacement[2]);
    }
    const float distance = sqrtf(
      displacement[0] * displacement[0] +
      displacement[1] * displacement[1] +
      displacement[2] * displacement[2]);
    const float inverse_distance = 1.0f / distance;
    const int type2 = type[atom2];
    const int atomic_number2 = zbl.atomic_numbers[type2];
    const float inverse_screening_length =
      (atomic_number_power1 + powf(static_cast<float>(atomic_number2), 0.23f)) *
      2.134563f;
    const float charge_product = K_C_SP * atomic_number1 * atomic_number2;
    float pair_energy = 0.0f;
    float pair_derivative = 0.0f;
    if (zbl.flexible) {
      const int lower_type = min(type1, type2);
      const int upper_type = max(type1, type2);
      const int pair = lower_type * zbl.num_types -
        lower_type * (lower_type - 1) / 2 + upper_type - lower_type;
      float parameters[10];
#pragma unroll
      for (int parameter = 0; parameter < 10; ++parameter) {
        parameters[parameter] = zbl.para[10 * pair + parameter];
      }
      find_f_and_fp_zbl(
        parameters,
        charge_product,
        inverse_screening_length,
        distance,
        inverse_distance,
        pair_energy,
        pair_derivative);
    } else {
      float inner_cutoff = zbl.rc_inner;
      float outer_cutoff = zbl.rc_outer;
      if (paramb.use_typewise_cutoff_zbl) {
        outer_cutoff = min(
          (COVALENT_RADIUS[atomic_number1 - 1] +
           COVALENT_RADIUS[atomic_number2 - 1]) *
            paramb.typewise_cutoff_zbl_factor,
          outer_cutoff);
        inner_cutoff = 0.0f;
      }
      find_f_and_fp_zbl(
        charge_product,
        inverse_screening_length,
        inner_cutoff,
        outer_cutoff,
        distance,
        inverse_distance,
        pair_energy,
        pair_derivative);
    }
    const float force_scale = 0.5f * pair_derivative * inverse_distance;
    const float pair_force[3] = {
      displacement[0] * force_scale,
      displacement[1] * force_scale,
      displacement[2] * force_scale};
    energy += 0.5f * pair_energy;
    if constexpr (SmallBox) {
      atomicAdd(&force_x[atom1], static_cast<double>(pair_force[0]));
      atomicAdd(&force_y[atom1], static_cast<double>(pair_force[1]));
      atomicAdd(&force_z[atom1], static_cast<double>(pair_force[2]));
      atomicAdd(&force_x[atom2], static_cast<double>(-pair_force[0]));
      atomicAdd(&force_y[atom2], static_cast<double>(-pair_force[1]));
      atomicAdd(&force_z[atom2], static_cast<double>(-pair_force[2]));
      atomicAdd(&virial[atom2], static_cast<double>(-displacement[0] * pair_force[0]));
      atomicAdd(&virial[atom_count + atom2], static_cast<double>(-displacement[1] * pair_force[1]));
      atomicAdd(&virial[2 * atom_count + atom2], static_cast<double>(-displacement[2] * pair_force[2]));
      atomicAdd(&virial[3 * atom_count + atom2], static_cast<double>(-displacement[0] * pair_force[1]));
      atomicAdd(&virial[4 * atom_count + atom2], static_cast<double>(-displacement[0] * pair_force[2]));
      atomicAdd(&virial[5 * atom_count + atom2], static_cast<double>(-displacement[1] * pair_force[2]));
      atomicAdd(&virial[6 * atom_count + atom2], static_cast<double>(-displacement[1] * pair_force[0]));
      atomicAdd(&virial[7 * atom_count + atom2], static_cast<double>(-displacement[2] * pair_force[0]));
      atomicAdd(&virial[8 * atom_count + atom2], static_cast<double>(-displacement[2] * pair_force[1]));
    } else {
      force_sum[0] += 2.0f * pair_force[0];
      force_sum[1] += 2.0f * pair_force[1];
      force_sum[2] += 2.0f * pair_force[2];
      virial_sum[0] -= displacement[0] * pair_force[0];
      virial_sum[1] -= displacement[1] * pair_force[1];
      virial_sum[2] -= displacement[2] * pair_force[2];
      virial_sum[3] -= displacement[0] * pair_force[1];
      virial_sum[4] -= displacement[0] * pair_force[2];
      virial_sum[5] -= displacement[1] * pair_force[2];
      virial_sum[6] -= displacement[1] * pair_force[0];
      virial_sum[7] -= displacement[2] * pair_force[0];
      virial_sum[8] -= displacement[2] * pair_force[1];
    }
  }
  potential[atom1] += energy;
  if constexpr (!SmallBox) {
    force_x[atom1] += force_sum[0];
    force_y[atom1] += force_sum[1];
    force_z[atom1] += force_sum[2];
#pragma unroll
    for (int component = 0; component < 9; ++component) {
      virial[component * atom_count + atom1] += virial_sum[component];
    }
  }
}

} // namespace

int NEP_Spin::Body_Channels::count(void) const
{
  return l_max_3body + static_cast<int>(has_q_222) + static_cast<int>(has_q_1111) +
         static_cast<int>(has_q_112) + static_cast<int>(has_q_123) +
         static_cast<int>(has_q_233) + static_cast<int>(has_q_134);
}

NEP_Spin::NEP_Spin(const char* file_potential, const int num_atoms) : num_atoms_(num_atoms)
{
  try {
    read_model(file_potential);
  } catch (const std::exception& error) {
    PRINT_INPUT_ERROR(error.what());
  }
  CHECK(gpuStreamCreate(&structural_descriptor_stream_));
  CHECK(gpuStreamCreate(&spin_descriptor_stream_));

  rc = std::max({model_.cutoff_radial, model_.cutoff_angular, model_.spin_cutoff});
  input_neighbor_capacity_ = model_.neighbor_capacity;
  resize_domain(num_atoms_);
  std::vector<float> spin_cutoff_pair(
    static_cast<std::size_t>(model_.num_types) * model_.num_types);
  for (int type_i = 0; type_i < model_.num_types; ++type_i) {
    for (int type_j = 0; type_j < model_.num_types; ++type_j) {
      spin_cutoff_pair[type_i * model_.num_types + type_j] = static_cast<float>(
        0.5 * (model_.spin_cutoff_by_type[type_i] + model_.spin_cutoff_by_type[type_j]));
    }
  }
  data_.spin_cutoff_pair.resize(spin_cutoff_pair.size());
  data_.spin_cutoff_pair.copy_from_host(spin_cutoff_pair.data(), spin_cutoff_pair.size());

  printf(
    "Use NEP4 Spin3 potential with %d atom type%s.\n",
    model_.num_types,
    model_.num_types == 1 ? "" : "s");
  printf(
    "    structural/spin descriptor dimensions = %d/%d.\n",
    model_.struct_descriptor_dim,
    model_.spin_descriptor_dim);
  printf(
    "    radial/angular/spin cutoffs = %g/%g/%g A.\n",
    model_.cutoff_radial,
    model_.cutoff_angular,
    model_.spin_cutoff);
}

void NEP_Spin::resize_domain(int num_atoms)
{
  num_atoms_ = num_atoms;
  small_box_initialized_ = false;
  neighbor_.initialize(rc, num_atoms_, input_neighbor_capacity_);
  model_.neighbor_capacity = neighbor_.capacity();
  const std::size_t slots =
    static_cast<std::size_t>(num_atoms_) * static_cast<std::size_t>(model_.neighbor_capacity);
  data_.NN_radial.resize_with_capacity(num_atoms_);
  data_.NN_angular.resize_with_capacity(num_atoms_);
  data_.NN_spin.resize_with_capacity(num_atoms_);
  data_.NL_radial.resize_with_capacity(slots);
  data_.NL_angular.resize_with_capacity(slots);
  data_.NL_spin.resize_with_capacity(slots);
  data_.descriptor.resize_with_capacity(static_cast<std::size_t>(num_atoms_) * model_.descriptor_dim);
  data_.Fp.resize_with_capacity(static_cast<std::size_t>(num_atoms_) * model_.descriptor_dim);
  data_.f12x.resize_with_capacity(slots);
  data_.f12y.resize_with_capacity(slots);
  data_.f12z.resize_with_capacity(slots);
  const std::size_t N = static_cast<std::size_t>(num_atoms_);
  const std::size_t spin3_state_size =
    N * static_cast<std::size_t>(model_.spin_polynomial_layout.moment_count);
  data_.spin3_moments.resize_with_capacity(spin3_state_size);
  const std::size_t sum_fxyz_size =
    N * static_cast<std::size_t>(model_.n_max_angular + 1) *
    static_cast<std::size_t>(
      (model_.body.l_max_3body + 1) * (model_.body.l_max_3body + 1) - 1);
  data_.sum_fxyz.resize_with_capacity(std::max<std::size_t>(sum_fxyz_size, 1));
}

NEP_Spin::~NEP_Spin(void)
{
  if (structural_descriptor_stream_ != nullptr) {
    CHECK(gpuStreamDestroy(structural_descriptor_stream_));
  }
  if (spin_descriptor_stream_ != nullptr) {
    CHECK(gpuStreamDestroy(spin_descriptor_stream_));
  }
}

void NEP_Spin::read_model(const char* file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    throw std::runtime_error(std::string("cannot open spin potential: ") + file_potential);
  }

  std::vector<std::string> tokens = next_tokens(input);
  const bool has_zbl = !tokens.empty() && tokens[0] == "nep4_spin3_zbl";
  if (tokens.size() < 3 ||
      (tokens[0] != "nep4_spin3" && !has_zbl)) {
    throw std::runtime_error("NEP_Spin accepts only nep4_spin3[/zbl] models");
  }
  zbl_.enabled = has_zbl;
  model_.num_types = parse_int(tokens[1]);
  if (
    model_.num_types < 1 || model_.num_types > NUM_ELEMENTS ||
    tokens.size() != static_cast<std::size_t>(model_.num_types + 2)) {
    throw std::runtime_error("invalid NEP_Spin model type count");
  }
  model_.elements.assign(tokens.begin() + 2, tokens.end());
  check_unique_elements(model_.elements);
  zbl_.num_types = model_.num_types;
  for (int type = 0; type < model_.num_types; ++type) {
    for (int element = 0; element < NUM_ELEMENTS; ++element) {
      if (model_.elements[type] == kElementSymbols[element]) {
        zbl_.atomic_numbers[type] = element + 1;
        break;
      }
    }
    if (zbl_.atomic_numbers[type] == 0) {
      throw std::runtime_error("unknown element in NEP_Spin ZBL model");
    }
  }

  tokens = next_tokens(input);
  require_line(tokens, "spin_mode", 3);
  if (parse_int(tokens[1]) != model_.spin_mode) {
    throw std::runtime_error("spin_mode metadata does not match the model tag");
  }
  const int spin_header_lines = parse_int(tokens[2]);
  constexpr int minimum_spin_header_lines = 9;
  const int maximum_spin_header_lines = minimum_spin_header_lines + 2;
  if (spin_header_lines < minimum_spin_header_lines ||
      spin_header_lines > maximum_spin_header_lines) {
    throw std::runtime_error("counted spin header has an invalid line count");
  }

  bool seen_baseline = false;
  bool seen_basis_size = false;
  bool seen_l_max = false;
  bool seen_compress = false;
  bool seen_cutoff = false;
  bool seen_scaler = false;
  bool seen_dof = false;
  bool seen_env = false;
  bool seen_order = false;
  bool seen_soc = false;
  bool seen_projection_size = false;
  for (int line = 0; line < spin_header_lines; ++line) {
    tokens = next_tokens(input);
    if (tokens.empty()) {
      throw std::runtime_error("truncated counted spin header");
    }
    const std::string& name = tokens[0];
    if (name == "spin_baseline") {
      if (seen_baseline || tokens.size() != static_cast<std::size_t>(model_.num_types + 1)) {
        throw std::runtime_error("spin_baseline must occur once with one value per type");
      }
      seen_baseline = true;
      for (int type = 0; type < model_.num_types; ++type) {
        model_.spin_baseline.push_back(parse_double(tokens[type + 1]));
      }
    } else if (name == "spin_basis_size") {
      require_line(tokens, "spin_basis_size", 2);
      if (seen_basis_size)
        throw std::runtime_error("duplicate spin_basis_size");
      seen_basis_size = true;
      model_.spin_basis_size[0] = parse_int(tokens[1]);
    } else if (name == "spin_l_max") {
      require_line(tokens, "spin_l_max", 2);
      if (seen_l_max)
        throw std::runtime_error("duplicate spin_l_max");
      seen_l_max = true;
      model_.spin_l_max[0] = parse_int(tokens[1]);
    } else if (name == "spin_compress") {
      require_line(tokens, "spin_compress", 2);
      if (seen_compress)
        throw std::runtime_error("duplicate spin_compress");
      seen_compress = true;
      model_.spin_compress = parse_int(tokens[1]);
    } else if (name == "spin_cutoff") {
      if (seen_cutoff)
        throw std::runtime_error("duplicate spin_cutoff");
      seen_cutoff = true;
      if (tokens.size() != 2 &&
          tokens.size() != static_cast<std::size_t>(model_.num_types + 1)) {
        throw std::runtime_error(
          "spin_cutoff requires one value or one value per atom type");
      }
      model_.spin_cutoff_by_type.resize(model_.num_types);
      for (int type = 0; type < model_.num_types; ++type) {
        const std::size_t value_index = tokens.size() == 2 ? 1 : type + 1;
        model_.spin_cutoff_by_type[type] = parse_double(tokens[value_index]);
        if (!(model_.spin_cutoff_by_type[type] > 0.0) ||
            !std::isfinite(model_.spin_cutoff_by_type[type])) {
          throw std::runtime_error("spin_cutoff values must be finite and positive");
        }
      }
      model_.spin_cutoff = *std::max_element(
        model_.spin_cutoff_by_type.begin(), model_.spin_cutoff_by_type.end());
    } else if (name == "spin_order") {
      require_line(tokens, "spin_order", 2);
      if (seen_order)
        throw std::runtime_error("duplicate spin_order");
      seen_order = true;
      model_.spin_order = parse_int(tokens[1]);
    } else if (name == "spin_soc") {
      require_line(tokens, "spin_soc", 2);
      if (seen_soc)
        throw std::runtime_error("duplicate spin_soc");
      seen_soc = true;
      model_.spin_soc = parse_flag(tokens[1], "spin_soc");
    } else if (name == "spin_projection_size") {
      require_line(tokens, "spin_projection_size", 2);
      if (seen_projection_size)
        throw std::runtime_error("duplicate spin_projection_size");
      seen_projection_size = true;
      model_.spin_projection_size = parse_int(tokens[1]);
    } else if (name == "spin_scaler") {
      require_line(tokens, "spin_scaler", 2);
      if (seen_scaler || parse_int(tokens[1]) != 1)
        throw std::runtime_error("spin_scaler must occur once and equal 1");
      seen_scaler = true;
    } else if (name == "spin_dof_type") {
      if (seen_dof)
        throw std::runtime_error("duplicate spin_dof_type");
      seen_dof = true;
      model_.spin_dof_type_active =
        parse_active_types(tokens, model_.elements, "spin_dof_type");
    } else if (name == "spin_env_type") {
      if (seen_env)
        throw std::runtime_error("duplicate spin_env_type");
      seen_env = true;
      model_.spin_env_type_active =
        parse_active_types(tokens, model_.elements, "spin_env_type");
    } else {
      throw std::runtime_error("unknown or unsupported spin header line: " + name);
    }
  }
  if (
    !seen_baseline || !seen_basis_size || !seen_l_max || !seen_compress ||
    !seen_cutoff || !seen_scaler || !seen_order || !seen_soc ||
    !seen_projection_size) {
    throw std::runtime_error("counted spin header is missing a required line");
  }
  if (!seen_dof) {
    model_.spin_dof_type_active.assign(model_.num_types, 1);
  }
  if (!seen_env) {
    model_.spin_env_type_active = model_.spin_dof_type_active;
  }
  for (int type = 0; type < model_.num_types; ++type) {
    if (
      model_.spin_dof_type_active[type] != 0 &&
      model_.spin_env_type_active[type] == 0) {
      throw std::runtime_error("spin_dof_type must be a subset of spin_env_type");
    }
  }

  if (zbl_.enabled) {
    tokens = next_tokens(input);
    if ((tokens.size() != 3 && tokens.size() != 4) || tokens[0] != "zbl") {
      throw std::runtime_error(
        "nep4_spin3_zbl requires zbl rc_inner rc_outer [zbl_factor]");
    }
    zbl_.rc_inner = parse_float(tokens[1]);
    zbl_.rc_outer = parse_float(tokens[2]);
    if (zbl_.rc_inner == 0.0f && zbl_.rc_outer == 0.0f) {
      zbl_.flexible = true;
    } else {
      if (!(zbl_.rc_inner >= 0.0f && zbl_.rc_inner < zbl_.rc_outer)) {
        throw std::runtime_error("invalid spin3 ZBL inner/outer cutoffs");
      }
      if (tokens.size() == 4) {
        paramb_.use_typewise_cutoff_zbl = true;
        paramb_.typewise_cutoff_zbl_factor = parse_float(tokens[3]);
        if (paramb_.typewise_cutoff_zbl_factor < 0.5f) {
          throw std::runtime_error("spin3 ZBL typewise cutoff factor must be >= 0.5");
        }
      }
    }
  }

  tokens = next_tokens(input);
  require_line(tokens, "cutoff", 5);
  model_.cutoff_radial = parse_double(tokens[1]);
  model_.cutoff_angular = parse_double(tokens[2]);
  model_.max_neighbors_global = parse_int(tokens[3]);
  model_.max_neighbors_angular = parse_int(tokens[4]);

  tokens = next_tokens(input);
  require_line(tokens, "n_max", 3);
  model_.n_max_radial = parse_int(tokens[1]);
  model_.n_max_angular = parse_int(tokens[2]);

  tokens = next_tokens(input);
  require_line(tokens, "basis_size", 3);
  model_.basis_size_radial = parse_int(tokens[1]);
  model_.basis_size_angular = parse_int(tokens[2]);

  tokens = next_tokens(input);
  if (tokens.size() < 2 || tokens.size() > 8 || tokens[0] != "l_max") {
    throw std::runtime_error("invalid l_max line");
  }
  model_.body.l_max_3body = parse_int(tokens[1]);
  if (tokens.size() >= 3)
    model_.body.has_q_222 = parse_flag(tokens[2], "q_222", true);
  if (tokens.size() >= 4)
    model_.body.has_q_1111 = parse_flag(tokens[3], "q_1111");
  if (tokens.size() >= 5)
    model_.body.has_q_112 = parse_flag(tokens[4], "q_112");
  if (tokens.size() >= 6)
    model_.body.has_q_123 = parse_flag(tokens[5], "q_123");
  if (tokens.size() >= 7)
    model_.body.has_q_233 = parse_flag(tokens[6], "q_233");
  if (tokens.size() >= 8)
    model_.body.has_q_134 = parse_flag(tokens[7], "q_134");

  tokens = next_tokens(input);
  require_line(tokens, "ANN", 3);
  model_.hidden_neurons = parse_int(tokens[1]);
  if (parse_int(tokens[2]) != 0) {
    throw std::runtime_error("NEP_Spin supports only a single hidden ANN layer");
  }

  if (
    model_.cutoff_radial <= 0.0 || model_.cutoff_angular <= 0.0 ||
    model_.spin_cutoff <= 0.0 ||
    model_.max_neighbors_global <= 0 || model_.max_neighbors_angular <= 0) {
    throw std::runtime_error("cutoffs and neighbor capacities must be finite and positive");
  }
  if (model_.n_max_radial < 0 || model_.n_max_radial > 12 ||
      model_.n_max_angular < 0 || model_.n_max_angular > 8 ||
      model_.basis_size_radial < 0 || model_.basis_size_radial > 16 ||
      model_.basis_size_angular < 0 || model_.basis_size_angular > 12 ||
      model_.body.l_max_3body < 0 || model_.body.l_max_3body > 8 ||
      model_.hidden_neurons < 1 || model_.hidden_neurons > 120) {
    throw std::runtime_error("ordinary NEP_Spin shape is outside the supported range");
  }
  if ((model_.body.has_q_222 || model_.body.has_q_112) && model_.body.l_max_3body < 2)
    throw std::runtime_error("q_222/q_112 require l_max_3body >= 2");
  if (model_.body.has_q_1111 && model_.body.l_max_3body < 1)
    throw std::runtime_error("q_1111 requires l_max_3body >= 1");
  if ((model_.body.has_q_123 || model_.body.has_q_233) && model_.body.l_max_3body < 3)
    throw std::runtime_error("q_123/q_233 require l_max_3body >= 3");
  if (model_.body.has_q_134 && model_.body.l_max_3body < 4)
    throw std::runtime_error("q_134 requires l_max_3body >= 4");

  if (
    model_.spin_compress < 1 || model_.spin_compress > 9 ||
    model_.spin_basis_size[0] != 8 || model_.spin_l_max[0] < 0 ||
    model_.spin_l_max[0] > 2 || model_.spin_order < 1 ||
    model_.spin_order > 3 || (model_.spin_soc != 0 && model_.spin_soc != 1) ||
    model_.spin_projection_size != 4 * model_.spin_compress * model_.spin_compress) {
    throw std::runtime_error("Spin3 O/C descriptor shape is outside the supported range");
  }

  model_.struct_descriptor_dim =
    model_.n_max_radial + 1 + (model_.n_max_angular + 1) * model_.body.count();
  if ((model_.n_max_angular + 1) * model_.body.count() > 90) {
    throw std::runtime_error("structural angular descriptor dimension exceeds 90");
  }
  model_.spin_polynomial_layout = make_spin_polynomial_layout(
    model_.spin_compress, model_.spin_l_max[0], model_.spin_order, model_.spin_soc);
  model_.spin_descriptor_dim = model_.spin_polynomial_layout.descriptor_dim;
  model_.descriptor_dim = model_.struct_descriptor_dim + model_.spin_descriptor_dim;
  if (model_.descriptor_dim > MAX_DIM_SPIN) {
    throw std::runtime_error(
      "combined structural and spin descriptor dimension exceeds GPUMD MAX_DIM_SPIN");
  }

  const std::size_t types = static_cast<std::size_t>(model_.num_types);
  const std::size_t type_pairs = types * types;
  model_.ann_parameter_count =
    static_cast<std::size_t>(model_.descriptor_dim + 2) * model_.hidden_neurons * types + 1;
  model_.radial_parameter_count = static_cast<std::size_t>(model_.n_max_radial + 1) *
                                  (model_.basis_size_radial + 1) * type_pairs;
  model_.angular_parameter_count = static_cast<std::size_t>(model_.n_max_angular + 1) *
                                   (model_.basis_size_angular + 1) * type_pairs;
  model_.spin_parameter_count = static_cast<std::size_t>(model_.spin_compress) *
                                (model_.spin_basis_size[0] + 1) * type_pairs;
  model_.spin_projection_parameter_count =
    static_cast<std::size_t>(model_.spin_projection_size);
  model_.model_parameter_count = model_.ann_parameter_count + model_.radial_parameter_count +
                                 model_.angular_parameter_count + model_.spin_parameter_count +
                                 model_.spin_projection_parameter_count;

  std::vector<float> parameters;
  parameters.reserve(model_.model_parameter_count + model_.descriptor_dim);
  for (std::size_t i = 0; i < model_.model_parameter_count + model_.descriptor_dim; ++i) {
    tokens = next_tokens(input);
    if (tokens.size() != 1) {
      throw std::runtime_error("truncated or malformed NEP_Spin numeric payload");
    }
    parameters.push_back(parse_float(tokens[0]));
  }
  if (zbl_.flexible) {
    const int zbl_parameter_count =
      10 * model_.num_types * (model_.num_types + 1) / 2;
    for (int parameter = 0; parameter < zbl_parameter_count; ++parameter) {
      tokens = next_tokens(input);
      if (tokens.size() != 1) {
        throw std::runtime_error("truncated flexible ZBL payload in NEP_Spin model");
      }
      zbl_.para[parameter] = parse_float(tokens[0]);
    }
  }
  if (!next_tokens(input).empty()) {
    throw std::runtime_error("extra header or numeric payload after NEP_Spin model");
  }
  data_.parameters.resize(parameters.size());
  data_.parameters.copy_from_host(parameters.data());

  const std::size_t descriptor_offset = model_.ann_parameter_count;
  const std::size_t ordinary_count =
    model_.radial_parameter_count + model_.angular_parameter_count;
  std::vector<float> descriptor_parameters(ordinary_count + model_.spin_parameter_count);
  const std::size_t radial_basis =
    static_cast<std::size_t>(model_.n_max_radial + 1) * (model_.basis_size_radial + 1);
  const std::size_t angular_basis =
    static_cast<std::size_t>(model_.n_max_angular + 1) * (model_.basis_size_angular + 1);
  for (std::size_t pair = 0; pair < type_pairs; ++pair) {
    for (std::size_t basis = 0; basis < radial_basis; ++basis) {
      descriptor_parameters[pair * radial_basis + basis] =
        parameters[descriptor_offset + basis * type_pairs + pair];
    }
    for (std::size_t basis = 0; basis < angular_basis; ++basis) {
      descriptor_parameters[
        model_.radial_parameter_count + pair * angular_basis + basis] =
        parameters[
          descriptor_offset + model_.radial_parameter_count + basis * type_pairs + pair];
    }
  }
  std::copy(
    parameters.begin() + descriptor_offset + ordinary_count,
    parameters.begin() + descriptor_offset + ordinary_count + model_.spin_parameter_count,
    descriptor_parameters.begin() + ordinary_count);
  data_.descriptor_parameters_type_pair.resize(descriptor_parameters.size());
  data_.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());
  if (model_.spin_projection_parameter_count > 0) {
    const std::size_t projection_offset = descriptor_offset + ordinary_count +
      model_.spin_parameter_count;
    std::vector<float> projection_parameters(
      parameters.begin() + projection_offset,
      parameters.begin() + projection_offset + model_.spin_projection_parameter_count);
    data_.spin_projection_parameters.resize(projection_parameters.size());
    data_.spin_projection_parameters.copy_from_host(projection_parameters.data());
  }

  std::vector<float> spin_baseline(model_.spin_baseline.begin(), model_.spin_baseline.end());
  data_.spin_baseline.resize(spin_baseline.size());
  data_.spin_baseline.copy_from_host(spin_baseline.data());
  data_.spin_dof_type_active.resize(model_.spin_dof_type_active.size());
  data_.spin_dof_type_active.copy_from_host(model_.spin_dof_type_active.data());
  data_.spin_env_type_active.resize(model_.spin_env_type_active.size());
  data_.spin_env_type_active.copy_from_host(model_.spin_env_type_active.data());

  paramb_.num_types = model_.num_types;
  paramb_.num_types_sq = model_.num_types * model_.num_types;
  paramb_.n_max_radial = model_.n_max_radial;
  paramb_.n_max_angular = model_.n_max_angular;
  paramb_.basis_size_radial = model_.basis_size_radial;
  paramb_.basis_size_angular = model_.basis_size_angular;
  paramb_.L_max = model_.body.l_max_3body;
  paramb_.num_L = model_.body.count();
  paramb_.has_q_222 = model_.body.has_q_222;
  paramb_.has_q_1111 = model_.body.has_q_1111;
  paramb_.has_q_112 = model_.body.has_q_112;
  paramb_.has_q_123 = model_.body.has_q_123;
  paramb_.has_q_233 = model_.body.has_q_233;
  paramb_.has_q_134 = model_.body.has_q_134;
  paramb_.dim_angular = (model_.n_max_angular + 1) * model_.body.count();
  paramb_.num_c_radial = static_cast<int>(model_.radial_parameter_count);
  paramb_.MN_radial = model_.neighbor_capacity;
  paramb_.MN_angular = model_.neighbor_capacity;
  paramb_.rc_radial_max = model_.cutoff_radial;
  paramb_.rc_radial_max_inv = 1.0f / model_.cutoff_radial;
  for (int type = 0; type < model_.num_types; ++type) {
    paramb_.rc_radial[type] = model_.cutoff_radial;
    paramb_.rc_angular[type] = model_.cutoff_angular;
  }

  ann_.dim = model_.descriptor_dim;
  ann_.num_neurons1 = model_.hidden_neurons;
  ann_.num_para_ann = static_cast<int>(model_.ann_parameter_count);
  ann_.num_para = static_cast<int>(model_.model_parameter_count);
  float* pointer = data_.parameters.data();
  for (int type = 0; type < model_.num_types; ++type) {
    ann_.w0[type] = pointer;
    pointer += model_.hidden_neurons * model_.descriptor_dim;
    ann_.b0[type] = pointer;
    pointer += model_.hidden_neurons;
    ann_.w1[type] = pointer;
    pointer += model_.hidden_neurons;
  }
  ann_.b1 = pointer;
  ann_.c_type_pair = data_.descriptor_parameters_type_pair.data();
  ann_.q_scaler = data_.parameters.data() + model_.model_parameter_count;

  const double enlarged = std::ceil(static_cast<double>(model_.max_neighbors_global) * 1.25);
  if (enlarged > std::numeric_limits<int>::max()) {
    throw std::runtime_error("neighbor capacity is too large");
  }
  model_.neighbor_capacity = static_cast<int>(enlarged);
}

void NEP_Spin::compute(
  Box&,
  const GPU_Vector<int>&,
  const GPU_Vector<double>&,
  GPU_Vector<double>&,
  GPU_Vector<double>&,
  GPU_Vector<double>&)
{
  PRINT_INPUT_ERROR("NEP_Spin requires spin inputs.");
}

void NEP_Spin::initialize_small_box(const Box& box)
{
  const double volume = box.get_volume();
  const double thickness[3] = {
    volume / box.get_area(0), volume / box.get_area(1), volume / box.get_area(2)};
  expanded_box_.num_cells[0] =
    box.pbc_x ? static_cast<int>(std::ceil(2.0 * rc / thickness[0])) : 1;
  expanded_box_.num_cells[1] =
    box.pbc_y ? static_cast<int>(std::ceil(2.0 * rc / thickness[1])) : 1;
  expanded_box_.num_cells[2] =
    box.pbc_z ? static_cast<int>(std::ceil(2.0 * rc / thickness[2])) : 1;

  const std::size_t cell_count =
    static_cast<std::size_t>(expanded_box_.num_cells[0]) *
    static_cast<std::size_t>(expanded_box_.num_cells[1]) *
    static_cast<std::size_t>(expanded_box_.num_cells[2]);
  if (
    cell_count == 0 ||
    cell_count > (std::numeric_limits<std::size_t>::max() - 1) /
                   static_cast<std::size_t>(num_atoms_)) {
    PRINT_INPUT_ERROR("NEP_Spin small-box image count overflows size_t.");
  }
  const std::size_t capacity =
    static_cast<std::size_t>(num_atoms_) * cell_count - 1;
  if (capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("NEP_Spin small-box neighbor capacity exceeds int.");
  }
  const std::size_t plane_size = static_cast<std::size_t>(num_atoms_) * capacity;
  if (capacity != 0 && plane_size / capacity != static_cast<std::size_t>(num_atoms_)) {
    PRINT_INPUT_ERROR("NEP_Spin small-box slot count overflows size_t.");
  }
  if (plane_size > std::numeric_limits<std::size_t>::max() / (3 * sizeof(double))) {
    PRINT_INPUT_ERROR("NEP_Spin small-box displacement storage overflows size_t.");
  }
  if (plane_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("NEP_Spin small-box slot plane exceeds int indexing.");
  }
  const int new_capacity = static_cast<int>(capacity);
  const bool resize_storage =
    !small_box_initialized_ || small_box_capacity_ != new_capacity;
  small_box_capacity_ = new_capacity;

  expanded_box_.h[0] = box.cpu_h[0] * expanded_box_.num_cells[0];
  expanded_box_.h[3] = box.cpu_h[3] * expanded_box_.num_cells[0];
  expanded_box_.h[6] = box.cpu_h[6] * expanded_box_.num_cells[0];
  expanded_box_.h[1] = box.cpu_h[1] * expanded_box_.num_cells[1];
  expanded_box_.h[4] = box.cpu_h[4] * expanded_box_.num_cells[1];
  expanded_box_.h[7] = box.cpu_h[7] * expanded_box_.num_cells[1];
  expanded_box_.h[2] = box.cpu_h[2] * expanded_box_.num_cells[2];
  expanded_box_.h[5] = box.cpu_h[5] * expanded_box_.num_cells[2];
  expanded_box_.h[8] = box.cpu_h[8] * expanded_box_.num_cells[2];
  expanded_box_.h[9] =
    expanded_box_.h[4] * expanded_box_.h[8] - expanded_box_.h[5] * expanded_box_.h[7];
  expanded_box_.h[10] =
    expanded_box_.h[2] * expanded_box_.h[7] - expanded_box_.h[1] * expanded_box_.h[8];
  expanded_box_.h[11] =
    expanded_box_.h[1] * expanded_box_.h[5] - expanded_box_.h[2] * expanded_box_.h[4];
  expanded_box_.h[12] =
    expanded_box_.h[5] * expanded_box_.h[6] - expanded_box_.h[3] * expanded_box_.h[8];
  expanded_box_.h[13] =
    expanded_box_.h[0] * expanded_box_.h[8] - expanded_box_.h[2] * expanded_box_.h[6];
  expanded_box_.h[14] =
    expanded_box_.h[2] * expanded_box_.h[3] - expanded_box_.h[0] * expanded_box_.h[5];
  expanded_box_.h[15] =
    expanded_box_.h[3] * expanded_box_.h[7] - expanded_box_.h[4] * expanded_box_.h[6];
  expanded_box_.h[16] =
    expanded_box_.h[1] * expanded_box_.h[6] - expanded_box_.h[0] * expanded_box_.h[7];
  expanded_box_.h[17] =
    expanded_box_.h[0] * expanded_box_.h[4] - expanded_box_.h[1] * expanded_box_.h[3];
  const double determinant =
    expanded_box_.h[0] *
      (expanded_box_.h[4] * expanded_box_.h[8] -
       expanded_box_.h[5] * expanded_box_.h[7]) +
    expanded_box_.h[1] *
      (expanded_box_.h[5] * expanded_box_.h[6] -
       expanded_box_.h[3] * expanded_box_.h[8]) +
    expanded_box_.h[2] *
      (expanded_box_.h[3] * expanded_box_.h[7] -
       expanded_box_.h[4] * expanded_box_.h[6]);
  for (int index = 9; index < 18; ++index) {
    expanded_box_.h[index] /= determinant;
  }

  if (resize_storage) {
    data_.NL_radial.resize(plane_size);
    data_.NL_angular.resize(plane_size);
    data_.NL_spin.resize(plane_size);
    data_.r12_radial.resize(3 * plane_size);
    data_.r12_angular.resize(3 * plane_size);
    data_.r12_spin.resize(3 * plane_size);
    data_.f12x.resize(plane_size);
    data_.f12y.resize(plane_size);
    data_.f12z.resize(plane_size);
  }
  small_box_initialized_ = true;
}

void NEP_Spin::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  const GPU_Vector<double>& spin,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& mforce)
{
  const int N = type.size();
  if (
    N != num_atoms_ || position.size() != static_cast<std::size_t>(3 * N) ||
    spin.size() != static_cast<std::size_t>(3 * N)) {
    PRINT_INPUT_ERROR("NEP_Spin input sizes do not match the initialized atom count.");
  }
  if (!(box.pbc_x && box.pbc_y && box.pbc_z)) {
    PRINT_INPUT_ERROR("NEP_Spin requires periodic boundary conditions in all three directions.");
  }
  const double volume = box.get_volume();
  const double thickness_x = volume / box.get_area(0);
  const double thickness_y = volume / box.get_area(1);
  const double thickness_z = volume / box.get_area(2);
  const double minimum_safe_thickness = neighbor_.minimum_safe_box_thickness(rc);
  const bool is_small_box =
    (box.pbc_x && thickness_x < minimum_safe_thickness) ||
    (box.pbc_y && thickness_y < minimum_safe_thickness) ||
    (box.pbc_z && thickness_z < minimum_safe_thickness);

  constexpr int block_size = 64;
  const int grid_size = (N - 1) / block_size + 1;
  if (is_small_box) {
    initialize_small_box(box);
    int* gpu_required_capacity = nullptr;
    CHECK(gpuGetSymbolAddress(
      reinterpret_cast<void**>(&gpu_required_capacity), required_small_box_capacity));
    const int zero = 0;
    CHECK(gpuMemcpy(gpu_required_capacity, &zero, sizeof(int), gpuMemcpyHostToDevice));
    find_neighbor_list_spin_small_box<<<grid_size, block_size>>>(
      paramb_,
      static_cast<float>(model_.spin_cutoff),
      N,
      box,
      expanded_box_,
      small_box_capacity_,
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + 2 * N,
      data_.NN_radial.data(),
      data_.NL_radial.data(),
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      data_.NN_spin.data(),
      data_.NL_spin.data(),
      data_.r12_radial.data(),
      data_.r12_angular.data(),
      data_.r12_spin.data(),
      gpu_required_capacity);
    GPU_CHECK_KERNEL
    int required_capacity = 0;
    CHECK(gpuMemcpy(
      &required_capacity, gpu_required_capacity, sizeof(int), gpuMemcpyDeviceToHost));
    if (required_capacity > small_box_capacity_) {
      PRINT_INPUT_ERROR(
        ("NEP_Spin small-box neighbor capacity overflow: required " +
         std::to_string(required_capacity) + " slots but capacity is " +
         std::to_string(small_box_capacity_) + ".")
          .c_str());
    }
    const int plane_size = N * small_box_capacity_;
    find_structural_descriptor<<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      box,
      data_.NN_radial.data(),
      data_.NL_radial.data(),
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      data_.r12_radial.data(),
      data_.r12_angular.data(),
      plane_size,
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + 2 * N,
      data_.descriptor.data(),
      data_.sum_fxyz.data());
    GPU_CHECK_KERNEL
    launch_spin3_descriptors(
      model_, box, type, position, spin, data_, data_.r12_spin.data(), plane_size);
    find_potential<<<grid_size, block_size>>>(
      N,
      model_.descriptor_dim,
      model_.hidden_neurons,
      model_.num_types,
      type.data(),
      data_.descriptor.data(),
      data_.parameters.data(),
      data_.parameters.data() + model_.model_parameter_count,
      data_.spin_baseline.data(),
      potential.data(),
      data_.Fp.data());
    GPU_CHECK_KERNEL
    find_force_radial_small_box<true><<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      0,
      N,
      data_.NN_radial.data(),
      data_.NL_radial.data(),
      type.data(),
      data_.r12_radial.data(),
      data_.r12_radial.data() + plane_size,
      data_.r12_radial.data() + 2 * plane_size,
      data_.Fp.data(),
      force.data(),
      force.data() + N,
      force.data() + 2 * N,
      virial.data());
    GPU_CHECK_KERNEL
    find_force_angular_small_box<true><<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      0,
      N,
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      type.data(),
      data_.r12_angular.data(),
      data_.r12_angular.data() + plane_size,
      data_.r12_angular.data() + 2 * plane_size,
      data_.Fp.data(),
      data_.sum_fxyz.data(),
      force.data(),
      force.data() + N,
      force.data() + 2 * N,
      virial.data());
    GPU_CHECK_KERNEL
    launch_spin3_forces(
      model_, box, type, position, spin, data_, force, mforce, virial,
      data_.r12_spin.data(), plane_size);
    if (zbl_.enabled) {
      find_spin_zbl_force<true><<<grid_size, block_size>>>(
        paramb_,
        N,
        zbl_,
        box,
        data_.NN_radial.data(),
        data_.NL_radial.data(),
        type.data(),
        nullptr,
        nullptr,
        nullptr,
        data_.r12_radial.data(),
        data_.r12_radial.data() + plane_size,
        data_.r12_radial.data() + 2 * plane_size,
        force.data(),
        force.data() + N,
        force.data() + 2 * N,
        virial.data(),
        potential.data());
      GPU_CHECK_KERNEL
    }
    return;
  }

  neighbor_.find_neighbor_global(rc, box, type, position);
  constexpr int neighbor_block_size = 128;
  const int neighbor_grid_size = (N - 1) / neighbor_block_size + 1;
  find_local_neighbors_spin<<<neighbor_grid_size, neighbor_block_size>>>(
    N,
    box,
    static_cast<float>(model_.cutoff_radial * model_.cutoff_radial),
    static_cast<float>(model_.cutoff_angular * model_.cutoff_angular),
    static_cast<float>(model_.spin_cutoff * model_.spin_cutoff),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    neighbor_.NN.data(),
    neighbor_.NL.data(),
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    data_.NN_spin.data(),
    data_.NL_spin.data());
  GPU_CHECK_KERNEL

  find_structural_descriptor<<<
    grid_size, block_size, 0, structural_descriptor_stream_>>>(
    paramb_,
    ann_,
    N,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    data_.NN_angular.data(),
    data_.NL_angular.data(),
    nullptr,
    nullptr,
    0,
    type.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    data_.descriptor.data(),
    data_.sum_fxyz.data());
  GPU_CHECK_KERNEL

  launch_spin3_descriptors(
    model_, box, type, position, spin, data_, nullptr, 0, spin_descriptor_stream_);
#ifdef USE_HIP
  CHECK(hipStreamSynchronize(structural_descriptor_stream_));
  CHECK(hipStreamSynchronize(spin_descriptor_stream_));
#else
  CHECK(cudaStreamSynchronize(structural_descriptor_stream_));
  CHECK(cudaStreamSynchronize(spin_descriptor_stream_));
#endif
  constexpr int potential_lanes_per_atom = 4;
  constexpr int potential_block_size = 128;
  constexpr int potential_atoms_per_block =
    potential_block_size / potential_lanes_per_atom;
  const int potential_grid_size =
    (N - 1) / potential_atoms_per_block + 1;
  const std::size_t potential_shared_size =
    static_cast<std::size_t>(potential_atoms_per_block) *
    (model_.descriptor_dim + model_.hidden_neurons) * sizeof(float);
  constexpr std::size_t maximum_potential_shared_size = 48 * 1024;
  if (potential_shared_size <= maximum_potential_shared_size) {
    find_potential_warp<potential_lanes_per_atom>
      <<<potential_grid_size, potential_block_size, potential_shared_size>>>(
      N,
      model_.descriptor_dim,
      model_.hidden_neurons,
      model_.num_types,
      type.data(),
      data_.descriptor.data(),
      data_.parameters.data(),
      data_.parameters.data() + model_.model_parameter_count,
      data_.spin_baseline.data(),
      potential.data(),
      data_.Fp.data());
  } else {
    find_potential<<<grid_size, block_size>>>(
      N,
      model_.descriptor_dim,
      model_.hidden_neurons,
      model_.num_types,
      type.data(),
      data_.descriptor.data(),
      data_.parameters.data(),
      data_.parameters.data() + model_.model_parameter_count,
      data_.spin_baseline.data(),
      potential.data(),
      data_.Fp.data());
  }
  GPU_CHECK_KERNEL

  find_force_radial<<<grid_size, block_size>>>(
    paramb_,
    ann_,
    N,
    0,
    N,
    box,
    data_.NN_radial.data(),
    data_.NL_radial.data(),
    type.data(),
    position.data(),
    position.data() + N,
    position.data() + 2 * N,
    data_.Fp.data(),
    force.data(),
    force.data() + N,
    force.data() + 2 * N,
    virial.data());
  GPU_CHECK_KERNEL

  constexpr int angular_atoms_per_warp = 4;
  constexpr int angular_edge_lanes = 8;
  constexpr int angular_block_size = 128;
  constexpr int angular_centers_per_block =
    angular_block_size / 32 * angular_atoms_per_warp;
  const int angular_grid_size =
    (N - 1) / angular_centers_per_block + 1;
  const int angular_state_size =
    paramb_.dim_angular + (paramb_.n_max_angular + 1) * NUM_OF_ABC;
  const std::size_t angular_shared_size =
    static_cast<std::size_t>(angular_centers_per_block) * angular_state_size * sizeof(float);
  constexpr std::size_t maximum_angular_shared_size = 48 * 1024;
  const bool use_direct_angular_force =
    angular_shared_size <= maximum_angular_shared_size;
  if (use_direct_angular_force) {
    // The body-channel signature comes from l_max in the model input. Keep a
    // generic instance for every signature other than the common L4+q222 case.
    const bool l4_q222_only =
      paramb_.L_max == 4 && paramb_.has_q_222 && !paramb_.has_q_1111 &&
      !paramb_.has_q_112 && !paramb_.has_q_123 && !paramb_.has_q_233 &&
      !paramb_.has_q_134 && paramb_.num_L == 5;
    if (l4_q222_only) {
      launch_force_angular_warp<angular_atoms_per_warp, angular_edge_lanes, true>(
        paramb_,
        ann_,
        box,
        type,
        position,
        data_,
        force,
        virial,
        angular_grid_size,
        angular_block_size,
        angular_shared_size);
    } else {
      launch_force_angular_warp<angular_atoms_per_warp, angular_edge_lanes, false>(
        paramb_,
        ann_,
        box,
        type,
        position,
        data_,
        force,
        virial,
        angular_grid_size,
        angular_block_size,
        angular_shared_size);
    }
  } else {
    find_partial_force_angular<<<grid_size, block_size>>>(
      paramb_,
      ann_,
      N,
      0,
      N,
      box,
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + 2 * N,
      data_.Fp.data(),
      data_.sum_fxyz.data(),
      data_.f12x.data(),
      data_.f12y.data(),
      data_.f12z.data());
  }
  GPU_CHECK_KERNEL
  if (!use_direct_angular_force) {
    find_properties_many_body(
      box,
      data_.NN_angular.data(),
      data_.NL_angular.data(),
      data_.f12x.data(),
      data_.f12y.data(),
      data_.f12z.data(),
      position,
      force,
      virial);
  }

  launch_spin3_forces(model_, box, type, position, spin, data_, force, mforce, virial);
  if (zbl_.enabled) {
    find_spin_zbl_force<false><<<grid_size, block_size>>>(
      paramb_,
      N,
      zbl_,
      box,
      neighbor_.NN.data(),
      neighbor_.NL.data(),
      type.data(),
      position.data(),
      position.data() + N,
      position.data() + 2 * N,
      nullptr,
      nullptr,
      nullptr,
      force.data(),
      force.data() + N,
      force.data() + 2 * N,
      virial.data(),
      potential.data());
    GPU_CHECK_KERNEL
  }
}
