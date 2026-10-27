/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#pragma once
namespace {

using SimulationBox = Box;

#include "force/nep_spin/nep_spin3_common.cuh"

enum class SpinVirialMode : int {
  disabled,
  center_owned,
  neighbor_owned,
  center_and_neighbor_float_sink,
};

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

__global__ void mask_inactive_spin_mforce_training(
  const int atom_count,
  const int atom_stride,
  const int* __restrict__ types,
  const int* __restrict__ spin_type_active,
  double* __restrict__ mforce)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < atom_count && spin_type_active[types[atom]] == 0) {
    mforce[atom] = 0.0;
    mforce[atom_stride + atom] = 0.0;
    mforce[2 * atom_stride + atom] = 0.0;
  }
}

#ifdef NEP_SPIN_LAYOUT_JIT
using SpinPolynomialLayout = SpinConstantLayout;
#else
using SpinPolynomialLayout = NEP_Spin_Model::Spin_Polynomial_Layout;
#endif

template <int C, bool NeedDerivatives>
__device__ __forceinline__ void evaluate_spin3_edge_weights_b9_f32(
  const float spin_cutoff,
  const float dist,
  const int num_types,
  const int type_pair,
  const float* descriptor_coefficients,
  const int spin_coefficient_offset,
  float* weights,
  float* weight_derivatives)
{
  constexpr int basis_count = 9;
  constexpr float pi = 3.14159265358979323846f;
  const int pair_count = num_types * num_types;
  const float inverse_cutoff = 1.0f / spin_cutoff;
  const float scaled_distance = dist * inverse_cutoff;
  const float phase = pi * scaled_distance;
  const float cutoff = 0.5f * cosf(phase) + 0.5f;
  const float shifted = scaled_distance - 1.0f;
  const float x = 2.0f * shifted * shifted - 1.0f;
  const float dxdr = 2.0f * shifted * inverse_cutoff;
  const float cutoff_derivative =
    NeedDerivatives ? -0.5f * pi * sinf(phase) * inverse_cutoff : 0.0f;
  float basis[basis_count] = {};
  float basis_derivative[basis_count] = {};
  basis[0] = cutoff;
  basis_derivative[0] = cutoff_derivative;
  const float raw1 = 0.5f * (x + 1.0f);
  basis[1] = raw1 * cutoff;
  basis_derivative[1] = dxdr * cutoff + raw1 * cutoff_derivative;
  float t0 = 1.0f;
  float t1 = x;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  for (int n = 2; n < basis_count; ++n) {
    const float t2 = 2.0f * x * t1 - t0;
    const float raw = 0.5f * (t2 + 1.0f);
    basis[n] = raw * cutoff;
    basis_derivative[n] =
      static_cast<float>(n) * u1 * dxdr * cutoff + raw * cutoff_derivative;
    const float u2 = 2.0f * x * u1 - u0;
    t0 = t1;
    t1 = t2;
    u0 = u1;
    u1 = u2;
  }
  for (int channel = 0; channel < C; ++channel) {
    float weight = 0.0f;
    float derivative = 0.0f;
    for (int basis_index = 0; basis_index < basis_count; ++basis_index) {
      const int index = spin_coefficient_offset +
        (channel * basis_count + basis_index) * pair_count + type_pair;
      weight += basis[basis_index] * descriptor_coefficients[index];
      if (NeedDerivatives) {
        derivative += basis_derivative[basis_index] * descriptor_coefficients[index];
      }
    }
    weights[channel] = weight;
    if (NeedDerivatives) {
      weight_derivatives[channel] = derivative;
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
  const int* types,
  const double* positions,
  const double* slot_r12,
  const int r12_plane_size,
  const int slot_index,
  const double* spins,
  const float* descriptor_coefficients,
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
      atom, neighbor, atom_stride, box, positions, rhat, dist);
  }
  for (int component = 0; component < 3; ++component) {
    si[component] = static_cast<float>(spins[component * atom_stride + atom]);
    sj[component] = static_cast<float>(spins[component * atom_stride + neighbor]);
  }
  if (!(dist > 1.0e-12f && dist < spin_cutoff) || spin_basis_size != 8) {
    return false;
  }
  const int type_pair = types[atom] * num_types + types[neighbor];
  const float resolved_spin_cutoff =
    spin_cutoff_pair == nullptr ? spin_cutoff : spin_cutoff_pair[type_pair];
  if (!(dist > 1.0e-12f && dist < resolved_spin_cutoff)) {
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

#include "force/nep_spin/nep_spin3_layout.cuh"
#include "force/nep_spin/nep_spin3_descriptor.cuh"
#include "force/nep_spin/nep_spin3_force.cuh"

template <int C, typename Layout>
void launch_spin3_descriptor(
  Layout layout,
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  float spin_cutoff,
  const float* spin_cutoff_pair,
  const int* type,
  const int* spin_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  const float* projection,
  float* moments,
  float* descriptors)
{
  constexpr int block_size = 128;
  Box box{};
  const int density_work = C * atom_count;
  build_spin3_oc_density_bank<C>
    <<<(density_work + block_size - 1) / block_size, block_size>>>(
      layout,
      atom_count,
      atom_count,
      struct_dim,
      num_types,
      spin_basis_size,
      spin_cutoff,
      spin_cutoff_pair,
      spin_coefficient_offset,
      box,
      type,
      spin_type_active,
      position,
      r12_spin,
      r12_plane_size,
      spin,
      NN_spin,
      NL_spin,
      descriptor_coefficients,
      descriptors,
      moments);
  contract_spin3_oc_descriptors<C>
    <<<(atom_count + block_size - 1) / block_size, block_size>>>(
      layout,
      atom_count,
      atom_count,
      struct_dim,
      type,
      spin_type_active,
      spin,
      projection,
      moments,
      descriptors);
  GPU_CHECK_KERNEL
}

template <int C, typename Layout>
void launch_spin3_force(
  Layout layout,
  int atom_count,
  int struct_dim,
  int num_types,
  int spin_basis_size,
  float spin_cutoff,
  const float* spin_cutoff_pair,
  const int* type,
  const int* spin_type_active,
  const double* position,
  const double* spin,
  const int* NN_spin,
  const int* NL_spin,
  const double* r12_spin,
  int r12_plane_size,
  const float* Fp,
  const float* descriptor_coefficients,
  int spin_coefficient_offset,
  const float* projection,
  const float* moments,
  double* force,
  double* mforce,
  double* virial)
{
  using Tile = Spin3ForceTile<C>;
  const int grid_size =
    (atom_count + Tile::centers_per_block - 1) / Tile::centers_per_block;
  const std::size_t shared_size =
    static_cast<std::size_t>(Tile::centers_per_block) * layout.moment_count * sizeof(float);
  Box box{};
  accumulate_spin3_oc_native_forces<
    C, SpinVirialMode::neighbor_owned, false, Tile::atoms_per_warp><<<
      grid_size, Tile::block_size, shared_size>>>(
        layout,
        atom_count,
        atom_count,
        struct_dim,
        num_types,
        spin_basis_size,
        spin_cutoff,
        spin_cutoff_pair,
        box,
        type,
        spin_type_active,
        position,
        r12_spin,
        r12_plane_size,
        spin,
        NN_spin,
        NL_spin,
        Fp,
        descriptor_coefficients,
        projection,
        moments,
        spin_coefficient_offset,
        force,
        mforce,
        virial,
        nullptr,
        nullptr);
  GPU_CHECK_KERNEL
}

__global__ void add_spin_training_outputs(
  int atom_count,
  const double* spin_force,
  const double* spin_mforce,
  const double* spin_virial,
  float* force,
  float* mforce,
  float* virial)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  for (int component = 0; component < 3; ++component) {
    const int index = component * atom_count + atom;
    force[index] += static_cast<float>(spin_force[index]);
    mforce[index] = static_cast<float>(spin_mforce[index]);
  }
  constexpr int source_component[6] = {0, 1, 2, 3, 5, 7};
  for (int component = 0; component < 6; ++component) {
    virial[component * atom_count + atom] += static_cast<float>(
      spin_virial[source_component[component] * atom_count + atom]);
  }
}

} // namespace
