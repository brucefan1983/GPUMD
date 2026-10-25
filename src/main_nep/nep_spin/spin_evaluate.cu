/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#include "nep_spin.cuh"
#include "main_nep/dataset.cuh"
#include "spin_parameters.cuh"
#include "model/box.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>

namespace {

using SimulationBox = Box;

#include "force/nep_spin3_common.cuh"

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

using SpinPolynomialLayout = NEP_Spin_Model::Spin_Polynomial_Layout;

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
      if constexpr (NeedDerivatives) {
        derivative += basis_derivative[basis_index] * descriptor_coefficients[index];
      }
    }
    weights[channel] = weight;
    if constexpr (NeedDerivatives) {
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

#include "force/nep_spin3_layout.cuh"
#include "force/nep_spin3_descriptor.cuh"
#include "force/nep_spin3_force.cuh"

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
  const int* spin_dof_type_active,
  const int* spin_env_type_active,
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
      spin_dof_type_active,
      spin_env_type_active,
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
      spin_dof_type_active,
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
  const int* spin_dof_type_active,
  const int* spin_env_type_active,
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
        spin_dof_type_active,
        spin_env_type_active,
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

void NEP_Spin_Model::initialize_spin(Spin_Parameters& para, int N, int device_count)
{
  spin_compress_ = para.spin_compress;
  spin_basis_size_ = para.spin_basis_size[0];
  spin_l_max_ = para.spin_l_max[0];
  spin_order_ = para.spin_order;
  spin_soc_ = para.spin_soc;
  struct_dim_ = para.dim_struct;
  spin_coefficient_offset_ =
    para.number_of_variables_descriptor - para.number_of_variables_descriptor_spin;
  spin_coefficient_offset_ -= para.number_of_variables_spin_projection;
  spin_projection_offset_ =
    spin_coefficient_offset_ + para.number_of_variables_descriptor_spin;
  num_types_ = para.num_types;
  spin_cutoff_ = para.spin_cutoff;

  polynomial_layout_ = make_spin_polynomial_layout<Spin_Polynomial_Layout>(
    spin_compress_, spin_l_max_, spin_order_, spin_soc_);

  for (int device_id = 0; device_id < device_count; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    auto& data = spin_data_[device_id];
    data.spin_dof_type_active.resize(num_types_);
    data.spin_env_type_active.resize(num_types_);
    data.spin_dof_type_active.copy_from_host(para.spin_dof_type_active.data());
    data.spin_env_type_active.copy_from_host(para.spin_env_type_active.data());
    std::vector<float> spin_cutoff_pair(
      static_cast<std::size_t>(num_types_) * num_types_);
    for (int type_i = 0; type_i < num_types_; ++type_i) {
      for (int type_j = 0; type_j < num_types_; ++type_j) {
        spin_cutoff_pair[type_i * num_types_ + type_j] =
          0.5f * (para.spin_cutoff_by_type[type_i] + para.spin_cutoff_by_type[type_j]);
      }
    }
    data.spin_cutoff_pair.resize(spin_cutoff_pair.size());
    data.spin_cutoff_pair.copy_from_host(spin_cutoff_pair.data(), spin_cutoff_pair.size());
    data.spin3_moments.resize(N * polynomial_layout_.moment_count);
    data.force.resize(3 * N);
    data.mforce.resize(3 * N);
    data.virial.resize(9 * N);
  }
}

void NEP_Spin_Model::find_additional_descriptors(
  Spin_Parameters&, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  const int plane_size = dataset.N * magnetic_data.max_NN_spin;
  const float* projection = annmb[device_id].c + spin_projection_offset_;
#define LAUNCH_SPIN3_DESCRIPTOR(C) \
    launch_spin3_descriptor<C>( \
      polynomial_layout_, dataset.N, struct_dim_, num_types_, spin_basis_size_, \
      spin_cutoff_, data.spin_cutoff_pair.data(), dataset.type.data(), data.spin_dof_type_active.data(), \
      data.spin_env_type_active.data(), magnetic_data.r_spin.data(), magnetic_data.spin.data(), \
      magnetic_data.NN_spin.data(), magnetic_data.NL_spin.data(), magnetic_data.r12_spin.data(), \
      plane_size, annmb[device_id].c, spin_coefficient_offset_, projection, \
      data.spin3_moments.data(), nep_data[device_id].descriptors.data())
  switch (spin_compress_) {
    case 1: LAUNCH_SPIN3_DESCRIPTOR(1); break;
    case 2: LAUNCH_SPIN3_DESCRIPTOR(2); break;
    case 3: LAUNCH_SPIN3_DESCRIPTOR(3); break;
    case 4: LAUNCH_SPIN3_DESCRIPTOR(4); break;
    case 5: LAUNCH_SPIN3_DESCRIPTOR(5); break;
    case 6: LAUNCH_SPIN3_DESCRIPTOR(6); break;
    case 7: LAUNCH_SPIN3_DESCRIPTOR(7); break;
    case 8: LAUNCH_SPIN3_DESCRIPTOR(8); break;
    case 9: LAUNCH_SPIN3_DESCRIPTOR(9); break;
  }
#undef LAUNCH_SPIN3_DESCRIPTOR
}

void NEP_Spin_Model::initialize_additional_outputs(
  Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  data.force.fill(0.0);
  data.mforce.fill(0.0);
  data.virial.fill(0.0);
  magnetic_data.mforce.fill(0.0f);
}

void NEP_Spin_Model::find_additional_force(
  Spin_Parameters&, Dataset& dataset, int device_id)
{
  auto& data = spin_data_[device_id];
  auto& magnetic_data = magnetic(dataset);
  const int block_size = 128;
  const int grid_size = (dataset.N + block_size - 1) / block_size;
  const int plane_size = dataset.N * magnetic_data.max_NN_spin;
  const float* projection = annmb[device_id].c + spin_projection_offset_;
#define LAUNCH_SPIN3_FORCE(C) \
    launch_spin3_force<C>( \
      polynomial_layout_, dataset.N, struct_dim_, num_types_, spin_basis_size_, \
      spin_cutoff_, data.spin_cutoff_pair.data(), dataset.type.data(), data.spin_dof_type_active.data(), \
      data.spin_env_type_active.data(), magnetic_data.r_spin.data(), magnetic_data.spin.data(), \
      magnetic_data.NN_spin.data(), magnetic_data.NL_spin.data(), magnetic_data.r12_spin.data(), \
      plane_size, nep_data[device_id].Fp.data(), annmb[device_id].c, \
      spin_coefficient_offset_, projection, data.spin3_moments.data(), \
      data.force.data(), data.mforce.data(), \
      data.virial.data())
  switch (spin_compress_) {
    case 1: LAUNCH_SPIN3_FORCE(1); break;
    case 2: LAUNCH_SPIN3_FORCE(2); break;
    case 3: LAUNCH_SPIN3_FORCE(3); break;
    case 4: LAUNCH_SPIN3_FORCE(4); break;
    case 5: LAUNCH_SPIN3_FORCE(5); break;
    case 6: LAUNCH_SPIN3_FORCE(6); break;
    case 7: LAUNCH_SPIN3_FORCE(7); break;
    case 8: LAUNCH_SPIN3_FORCE(8); break;
    case 9: LAUNCH_SPIN3_FORCE(9); break;
  }
#undef LAUNCH_SPIN3_FORCE

  mask_inactive_spin_mforce_training<<<grid_size, block_size>>>(
    dataset.N,
    dataset.N,
    dataset.type.data(),
    data.spin_dof_type_active.data(),
    data.mforce.data());
  add_spin_training_outputs<<<grid_size, block_size>>>(
    dataset.N,
    data.force.data(),
    data.mforce.data(),
    data.virial.data(),
    dataset.force.data(),
    magnetic_data.mforce.data(),
    dataset.virial.data());
  GPU_CHECK_KERNEL
}
