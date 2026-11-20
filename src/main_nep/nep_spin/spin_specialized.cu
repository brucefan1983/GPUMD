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
  Chengbing Chen: Initial implementation. Reused forward geometry in the training adjoint.
------------------------------------------------------------------------------*/

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
#include "force/nep_spin/spin_kernels.cuh"
#include "force/nep_spin/spin_layout.cuh"
#include "force/nep_spin/spin_md_layouts.cuh"
#include "model/box.cuh"
#include "nep_special_config.cuh"
#include "nep_spin.cuh"
#include "utilities/gpu_macro.cuh"
#include <type_traits>

namespace
{
constexpr bool quadratic_training = nep_spin::CompiledLayout::channels == 3 &&
                                    nep_spin::CompiledLayout::lmax == 2 &&
                                    nep_spin::CompiledLayout::order == 2;
using QuadraticForceLayout = typename std::conditional<
  nep_spin::CompiledLayout::soc,
  nep_spin::MDLayoutSOC<4, true>,
  nep_spin::MDLayoutNoSOC<4, true>>::type;
using Compiled = nep_spin::CompiledLayout;
constexpr bool fused_polynomial = nep_spin::fused_training_adjoint(
  Compiled::channels, Compiled::lmax, Compiled::order, Compiled::soc);
struct WarpForceLayout : Compiled {
  enum { tile = nep_spin::SpinProjection::width };
};
using PolynomialForceLayout =
  typename std::conditional<fused_polynomial, WarpForceLayout, Compiled>::type;
using ForceLayout =
  typename std::conditional<quadratic_training, QuadraticForceLayout, PolynomialForceLayout>::type;
} // namespace

extern "C" void nep_spin_launch_descriptor_v4(
  int atom_count,
  int struct_dim,
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
  const float* projection,
  float* moments,
  float* descriptors,
  double* wide_spin_descriptors,
  const double* wide_radial,
  nep_spin::ProjectionView forward)
{
  nep_spin::CompiledLayout layout;
  if (layout.edge_cached && r12_plane_size > 0) {
    float* cache = const_cast<float*>(descriptor_coefficients) +
                   NUM_TYPES_JIT * NUM_TYPES_JIT * layout.channels * (BASIS_SIZE_ANGULAR_JIT + 1);
    layout.edge_radial = cache;
    nep_spin::cache_radial_edges<nep_spin::CompiledLayout, BASIS_SIZE_ANGULAR_JIT>
      <<<(r12_plane_size + 127) / 128, 128>>>(
        atom_count,
        r12_plane_size,
        NUM_TYPES_JIT,
        type,
        NN_spin,
        NL_spin,
        r12_spin,
        spin_cutoff_pair,
        descriptor_coefficients,
        cache);
    GPU_CHECK_KERNEL
  }
#ifdef NEP_SPIN_WIDE_FORWARD
  auto* radial = wide_radial;
  auto* q = wide_spin_descriptors;
  struct_dim = 0;
#else
  auto* radial = descriptor_coefficients;
  auto* q = descriptors;
#endif
  constexpr bool warp =
    nep_spin::warp_training_forward(Compiled::lmax, Compiled::order, Compiled::soc);
  constexpr int lanes = warp ? int(nep_spin::SpinProjection::width) : int(Compiled::tile);
  nep_spin::descriptors<warp, false, warp>
    <<<(atom_count * Compiled::channels * lanes + 63) / 64, 64>>>(
      layout,
      atom_count,
      struct_dim,
      NUM_TYPES_JIT,
      N_MAX_ANGULAR_JIT,
      BASIS_SIZE_ANGULAR_JIT,
      Box{},
      type,
      spin_type_active,
      position,
      spin,
      NN_spin,
      NL_spin,
      r12_spin,
      r12_plane_size,
      spin_cutoff_pair,
      radial,
      projection,
      moments,
      q,
      forward,
      fused_polynomial ? moments + size_t(atom_count) * Compiled::channels * Compiled::components
                       : nullptr);
  GPU_CHECK_KERNEL
}

extern "C" void nep_spin_launch_force_v5(
  int atom_count,
  int struct_dim,
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
  const float* projection,
  const float* moments,
  double* force,
  double* mforce,
  double* virial,
  const double* wide_radial,
  nep_spin::ProjectionView transpose)
{
  ForceLayout layout;
  if (layout.edge_cached)
    layout.edge_radial = descriptor_coefficients + NUM_TYPES_JIT * NUM_TYPES_JIT * layout.channels *
                                                     (BASIS_SIZE_ANGULAR_JIT + 1);
#ifdef NEP_SPIN_WIDE_FORWARD
  const auto* radial = wide_radial;
#else
  const auto* radial = descriptor_coefficients;
#endif
  // Retain training's neighbor-owned virial while sharing the MD polynomial adjoint.
  constexpr bool fuse = quadratic_training || fused_polynomial;
  constexpr int batches = fuse ? (Compiled::channels + 2) / 3 : int(Compiled::channels);
  nep_spin::forces<ForceLayout, false, quadratic_training, fuse, fused_polynomial, false, true>
    <<<(atom_count * batches * ForceLayout::tile + 63) / 64, 64>>>(
      layout,
      atom_count,
      struct_dim,
      NUM_TYPES_JIT,
      N_MAX_ANGULAR_JIT,
      BASIS_SIZE_ANGULAR_JIT,
      Box{},
      type,
      spin_type_active,
      position,
      spin,
      NN_spin,
      NL_spin,
      r12_spin,
      r12_plane_size,
      spin_cutoff_pair,
      radial,
      projection,
      moments,
      Fp,
      force,
      mforce,
      virial,
      true,
      nullptr,
      transpose);
  GPU_CHECK_KERNEL
}

extern "C" int nep_train_specialized_interface_version()
{
  return NEP_SPECIAL_CONFIG_INTERFACE_VERSION;
}
#endif
