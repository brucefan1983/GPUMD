/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
#include "nep_special_config.cuh"
#include "nep_spin.cuh"
#include "model/box.cuh"
#include "utilities/gpu_macro.cuh"
#include "spin_training_kernels.cuh"

extern "C" void nep_spin_launch_descriptor(
  NEP_Spin_Model::Spin_Polynomial_Layout,
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
  launch_spin3_descriptor<SpinConstantLayout::channels>(
    SpinConstantLayout{}, atom_count, struct_dim, num_types, spin_basis_size,
    spin_cutoff, spin_cutoff_pair, type, spin_type_active,
    position, spin, NN_spin, NL_spin, r12_spin, r12_plane_size,
    descriptor_coefficients, spin_coefficient_offset, projection, moments, descriptors);
}

extern "C" void nep_spin_launch_force(
  NEP_Spin_Model::Spin_Polynomial_Layout,
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
  launch_spin3_force<SpinConstantLayout::channels>(
    SpinConstantLayout{}, atom_count, struct_dim, num_types, spin_basis_size,
    spin_cutoff, spin_cutoff_pair, type, spin_type_active,
    position, spin, NN_spin, NL_spin, r12_spin, r12_plane_size,
    Fp, descriptor_coefficients, spin_coefficient_offset, projection, moments,
    force, mforce, virial);
}

extern "C" int nep_train_specialized_interface_version() { return NEP_SPECIAL_CONFIG_INTERFACE_VERSION; }
#endif
