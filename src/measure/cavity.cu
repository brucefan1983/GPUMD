/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details. You should have received a copy of the GNU General
   Public License along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Dump energy/force/virial with all loaded potentials at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "cavity.cuh"
#include "force/nep3.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <vector>


const double BOHR_IN_ANGSTROM = 0.529177249;

static __global__ void sum_dipole(
  const int N, const int number_of_patches, const double* g_virial_per_atom, double* g_dipole)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ double s_d[1024];
  double d = 0.0;

  const unsigned int componentIdx = blockIdx.x * N;

  // 1024 threads, each summing a patch of N/1024 atoms
  for (int patch = 0; patch < number_of_patches; ++patch) {
    int atomIdx = tid + patch * 1024;
    if (atomIdx < N)
      d += g_virial_per_atom[componentIdx + atomIdx];
  }

  // save the sum for this patch
  s_d[tid] = d;
  __syncthreads();

  // aggregate the patches in parallel
#pragma unroll
  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_d[tid] += s_d[tid + offset];
    }
    __syncthreads();
  }
  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_d[tid] += s_d[tid + offset];
    }
    __syncwarp();
  }

  // save the final value
  if (tid == 0) {
    g_dipole[bid] = s_d[0];
  }
}


static __global__ void get_center_of_mass(
  const int N, 
  const int number_of_patches, 
  const double total_mass,
  const double* g_mass_per_atom,  
  const double* g_position_per_atom,
  double* g_center_of_mass)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ double s_d[1024];
  double d = 0.0;

  const unsigned int componentIdx = bid * N;

  // 1024 threads, each summing a patch of N/1024 atoms
  for (int patch = 0; patch < number_of_patches; ++patch) {
    int atomIdx = tid + patch * 1024;
    if (atomIdx < N) {
      d += g_mass_per_atom[atomIdx] * g_position_per_atom[componentIdx + atomIdx];
    }
  }

  // save the sum for this patch
  s_d[tid] = d;
  __syncthreads();

  // aggregate the patches in parallel
  #pragma unroll
  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_d[tid] += s_d[tid + offset];
    }
    __syncthreads();
  }
  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_d[tid] += s_d[tid + offset];
    }
    __syncwarp();
  }

  // save the final value
  if (tid == 0) {
    g_center_of_mass[bid] = s_d[0] / total_mass;
  }
}

static __global__ void copy_atomic_properties(
  int N,
  double* g_mass,
  double* ref_g_mass,
  double* g_p,
  double* ref_g_p)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    // Copy the masses
    g_mass[n1] = ref_g_mass[n1];

    // Copy the atom positions
    g_p[n1 + 0 * N] = ref_g_p[n1 + 0 * N];
    g_p[n1 + 1 * N] = ref_g_p[n1 + 1 * N];
    g_p[n1 + 2 * N] = ref_g_p[n1 + 2 * N];
  }
}


static __global__ void initialize_properties(
  int N,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_pe,
  double* g_virial,
  double* g_virial_sum)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    g_virial[n1 + 0 * N] = 0.0;
    g_virial[n1 + 1 * N] = 0.0;
    g_virial[n1 + 2 * N] = 0.0;
    g_virial[n1 + 3 * N] = 0.0;
    g_virial[n1 + 4 * N] = 0.0;
    g_virial[n1 + 5 * N] = 0.0;
    g_virial[n1 + 6 * N] = 0.0;
    g_virial[n1 + 7 * N] = 0.0;
    g_virial[n1 + 8 * N] = 0.0;
  }
  if (n1 == 0) {
    // Only need to set g_virial_sum to zero once
    g_virial_sum[0] = 0.0;
    g_virial_sum[1] = 0.0;
    g_virial_sum[2] = 0.0;
  }
}

static __global__ void displace_atom(
    int index,
    double* g_position,
    double displacement)
{
  // incredibly innefficient, do something else here
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 == 0) {
    // displacement should have the appropriate sign
    // depending on if it's forwards or backwards.
    g_position[index] += displacement;
  }
}

void Cavity::parse(const char** param, int num_param)
{
  enabled_ = true;
  printf("Cavity dynamics\n");

  if (num_param != 4) {
    PRINT_INPUT_ERROR("cavity should have 3 parameters.");
  }
  if (!is_valid_real(param[1], &coupling_strength)) {
    PRINT_INPUT_ERROR("coupling strength should be a real number.");
  }
  if (coupling_strength < 0.0) {
    PRINT_INPUT_ERROR("coupling strength cannot be negative.");
  }

  if (!is_valid_real(param[2], &cavity_frequency)) {
    PRINT_INPUT_ERROR("cavity frequency should be a real number.");
  }
  if (cavity_frequency < 0.0) {
    PRINT_INPUT_ERROR("cavity frequency cannot be negative.");
  }
  if (!is_valid_int(param[3], &charge)) {
    PRINT_INPUT_ERROR("total system charge should be an integer.");
  }
  printf("   coupling strength %f.\n", coupling_strength);
  printf("   cavity frequency %f.\n", cavity_frequency);
  printf("   total charge %d.\n", charge);
}

void Cavity::preprocess(
  const int number_of_atoms, 
  const int number_of_potentials, 
  Box& box,
  Atom& atom,
  Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  if (enabled_) {
    std::string filename_ = "jacobian.out";
    file_ = my_fopen(filename_.c_str(), "a");
    gpu_dipole_.resize(3);
    gpu_dipole_jacobian_.resize(number_of_atoms * 3 * 3);
    cpu_dipole_.resize(3);
    cpu_dipole_jacobian_.resize(number_of_atoms * 3 * 3);

    // Set up a local copy of the Atoms, on which to compute the dipole
    // Typically in GPUMD we are limited by computational speed, not memory,
    // so we can sacrifice a bit of memory to skip having to recompute the forces
    // & virials with the original potential
    atom_copy.number_of_atoms = number_of_atoms;
    atom_copy.type.resize(number_of_atoms);
    atom_copy.mass.resize(number_of_atoms);
    atom_copy.position_per_atom.resize(number_of_atoms * 3);
    atom_copy.force_per_atom.resize(number_of_atoms * 3);
    atom_copy.virial_per_atom.resize(number_of_atoms * 9);
    atom_copy.potential_per_atom.resize(number_of_atoms);

    // make sure that the second potential is actually a dipole model.
    if (number_of_potentials != 2) {
      PRINT_INPUT_ERROR("cavity requires two potentials to be specified.");
    }
    // Multiple potentials may only be used with NEPs, so we know that
    // the second potential must be an NEP
    if (force.potentials[1]->nep_model_type != 1) {
      PRINT_INPUT_ERROR("cavity requires the second NEP potential to be a dipole model.");
    }

    // Copy the mass array on atoms to the CPU
    // and compute the total mass. Do this on the CPU
    // since we only need to do it once
    masses_.resize(number_of_atoms);
    for (int i=0; i<number_of_atoms; i++) {
      double m_i = atom.cpu_mass[i];
      masses_[i] = m_i;
      mass_ += m_i;
    }
    // Transfer the cpu_types to our copy of the Atoms object
    atom_copy.type.copy_from_host(atom.cpu_type.data());

    // initialize the cavity stuff
    // initial cavity coordinate is equal to
    // self._cav_q0 = self.coupling_strength_v @ dipole_v / self.cav_frequency
    // so we need the dipole initially

    const int number_of_atoms = atom_copy.number_of_atoms;
    // copy stuff to our atoms object
    copy_atomic_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms,
      atom_copy.mass.data(),
      atom.mass.data(),
      atom_copy.position_per_atom.data(),
      atom.position_per_atom.data());
    CUDA_CHECK_KERNEL

    get_dipole(box, force, gpu_dipole_);
    gpu_dipole_.copy_to_host(cpu_dipole_.data());
    // For now, only allow a coupling strength vector in the z-direction.
    q0 = coupling_strength * cpu_dipole_[2] / cavity_frequency;
    std::cout << "init: " << mass_ << " " << q0 << "\n";
  }
}

void Cavity::process(
  int step,
  const double global_time,
  const int number_of_atoms_fixed,
  std::vector<Group>& group,
  Box& box,
  Atom& atom,
  Force& force)
{
  if (!enabled_) {
    return;
  }
  const int number_of_atoms = atom_copy.number_of_atoms;
  // copy stuff to our atoms object
  copy_atomic_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom_copy.mass.data(),
    atom.mass.data(),
    atom_copy.position_per_atom.data(),
    atom.position_per_atom.data());
  CUDA_CHECK_KERNEL

  // Compute the dipole
  // Consider this 
  /* if self.gpumddipole:
      atoms_copy = atoms.copy()
      atoms_copy.set_positions(atoms.get_positions() - atoms.get_center_of_mass())
      gpumd_dipole = (self.calcdipole.get_dipole_moment(atoms_copy) * Bohr +
                      self.charge * atoms.get_center_of_mass())
  */
  get_dipole(box, force, gpu_dipole_);
  gpu_dipole_.copy_to_host(cpu_dipole_.data());
  // The dipole is currently in atomic units.
  // Convert it to GPUMD units (Bohr -> Å),
  // and subtract the charge times the COM.
  GPU_Vector<double> gpu_center_of_mass(3);
  std::vector<double> cpu_center_of_mass(3);
  _get_center_of_mass(gpu_center_of_mass);
  gpu_center_of_mass.copy_to_host(cpu_center_of_mass.data());
  
  for (int i = 0; i < 3; i++){
    cpu_dipole_[i] *= BOHR_IN_ANGSTROM;
    cpu_dipole_[i] += charge * cpu_center_of_mass[i];
  }
  // Compute the dipole jacobian
  // The dipole jacobian has already been converted from atomic
  // units to GPUMD units and shifted appropriately.
  get_dipole_jacobian(box, force, number_of_atoms, 0.01, charge);

  // Update the cavity position
  // TODO
  
  // Compute the cavity force
  // TODO

  // Dump things
  //gpu_dipole_jacobian_.copy_to_host(cpu_dipole_jacobian_.data());
  // Write properties
  write_dipole(step);
}


void Cavity::get_dipole(
  Box& box,
  Force& force,
  GPU_Vector<double>& dipole_) 
{
  const int number_of_atoms = atom_copy.number_of_atoms;
  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom_copy.force_per_atom.data(),
    atom_copy.force_per_atom.data() + number_of_atoms,
    atom_copy.force_per_atom.data() + number_of_atoms * 2,
    atom_copy.potential_per_atom.data(),
    atom_copy.virial_per_atom.data(),
    dipole_.data());
  CUDA_CHECK_KERNEL
  // Compute the dipole
  force.potentials[1]->compute(
    box,
    atom_copy.type,
    atom_copy.position_per_atom,
    atom_copy.potential_per_atom,
    atom_copy.force_per_atom,
    atom_copy.virial_per_atom);
  
  // Aggregate virial_per_atom into dipole
  const int number_of_threads = 1024;
  const int number_of_atoms_per_thread = (number_of_atoms - 1) / number_of_threads + 1;
  sum_dipole<<<3, 1024>>>(
    number_of_atoms,
    number_of_atoms_per_thread,
    atom_copy.virial_per_atom.data(),
    dipole_.data());
  CUDA_CHECK_KERNEL
}


void Cavity::_get_center_of_mass(GPU_Vector<double>& gpu_center_of_mass) {
  const int number_of_atoms = atom_copy.number_of_atoms;
  const int number_of_threads = 1024;
  const int number_of_atoms_per_thread = (number_of_atoms - 1) / number_of_threads + 1;
  get_center_of_mass<<<3, 1024>>>(
    number_of_atoms,
    number_of_atoms_per_thread,
    mass_,
    atom_copy.mass.data(),
    atom_copy.position_per_atom.data(),
    gpu_center_of_mass.data());
  CUDA_CHECK_KERNEL
}


void Cavity::get_dipole_jacobian(
  Box& box,
  Force& force,
  int number_of_atoms,
  double displacement, 
  double charge) 
{
  /**
   @brief Get dipole gradient through finite differences.
   @details Calculates the dipole gradient, a (N_atoms, 3, 3) tensor for the
   gradients dµ_k/dr_ij, for atom i, Cartesian direction j (x, y, z) and dipole
   moment component k.
   Before computing the gradient the dipoles are corrected using the center of
   mass and the total system charge, supplied via the parameter `charge`.
   @param displacement        Displacement in Å.
   @param charge              Total system charge, used to correct dipoles.
  */
  const int N_cartesian = 3;
  const int N_components = 3;
  const int values_per_atom = N_cartesian * N_components;

  // Second order central differences
  // Need to compute four dipoles for each structure, yielding an error O(h^4)
  // Coefficients are defined here:
  // https://en.wikipedia.org/wiki/Finite_difference_coefficient#Central_finite_difference

  // dipole vectors are zeroed in find_dipole, can be allocated here
  GPU_Vector<double> gpu_dipole_forward_one_h(3);
  GPU_Vector<double> gpu_dipole_forward_two_h(3);
  GPU_Vector<double> gpu_dipole_backward_one_h(3);
  GPU_Vector<double> gpu_dipole_backward_two_h(3);
  std::vector<double> dipole_forward_one_h(3);
  std::vector<double> dipole_forward_two_h(3);
  std::vector<double> dipole_backward_one_h(3);
  std::vector<double> dipole_backward_two_h(3);

  // use center of mass to correct for permanent dipole
  GPU_Vector<double> gpu_center_of_mass(3);
  std::vector<double> center_of_mass_forward_one_h(3);
  _get_center_of_mass(gpu_center_of_mass);
  gpu_center_of_mass.copy_to_host(center_of_mass_forward_one_h.data());

  std::vector<double> center_of_mass_forward_two_h(
      center_of_mass_forward_one_h);
  std::vector<double> center_of_mass_backward_one_h(
      center_of_mass_forward_one_h);
  std::vector<double> center_of_mass_backward_two_h(
      center_of_mass_forward_one_h);
  // Positions are in order [x1, ..., xN, y1, ..., yN, ...]
  int index = 0;
  double old_center_of_mass = 0.0;
  const double displacement_over_M = displacement / mass_;
  const double one_over_displacement =
      1.0 / displacement; // coefficients are scaled properly

  const double c0 = -1.0 / 12.0; // coefficient for 2h
  const double c1 = 2.0 / 3.0;   // coefficient for h
  for (int i = 0; i < N_cartesian; i++) {
    for (int j = 0; j < number_of_atoms; j++) {
      index = number_of_atoms * i + j; // idx of position to change
      old_center_of_mass = center_of_mass_forward_one_h[i];

      // --- Forward displacement
      // Step one displacement forward
      // I hate this
      displace_atom<<<1, 1>>>(
          index, 
          atom_copy.position_per_atom.data(), 
          displacement); // +h
      CUDA_CHECK_KERNEL
      center_of_mass_forward_one_h[i] +=
          displacement_over_M * masses_[j];   // center of mass gets moved by
                                              // +h/N in the same direction
      get_dipole(box, force, gpu_dipole_forward_one_h);
      gpu_dipole_forward_one_h.copy_to_host(dipole_forward_one_h.data());

      // Step two displacements forward
      displace_atom<<<1, 1>>>(
          index, 
          atom_copy.position_per_atom.data(), 
          displacement); // displaced +2h in total
      CUDA_CHECK_KERNEL
      center_of_mass_forward_two_h[i] +=
          2 * displacement_over_M * masses_[j]; // center of mass gest moved by
                                                // +2h/N in the same direction
      get_dipole(box, force, gpu_dipole_forward_two_h);
      gpu_dipole_forward_two_h.copy_to_host(dipole_forward_two_h.data());

      // --- Backwards displacement
      displace_atom<<<1, 1>>>(
          index, 
          atom_copy.position_per_atom.data(), 
          -3*displacement); // 2h - 3h = -h
      CUDA_CHECK_KERNEL
      center_of_mass_backward_one_h[i] -= displacement_over_M * masses_[j];
      get_dipole(box, force, gpu_dipole_backward_one_h);
      gpu_dipole_backward_one_h.copy_to_host(dipole_backward_one_h.data());

      displace_atom<<<1, 1>>>(
          index, 
          atom_copy.position_per_atom.data(), 
          -displacement); // -h - h = -2h
      CUDA_CHECK_KERNEL
      center_of_mass_backward_two_h[i] -=
          2 * displacement_over_M * masses_[j];
      get_dipole(box, force, gpu_dipole_backward_two_h);
      gpu_dipole_backward_two_h.copy_to_host(dipole_backward_two_h.data());

      for (int k = 0; k < N_components; k++) {
        cpu_dipole_jacobian_[i * N_components + j * values_per_atom + k] =
            (c0 * (dipole_forward_two_h[k] * BOHR_IN_ANGSTROM +
                   charge * center_of_mass_forward_two_h[k]) +
             c1 * (dipole_forward_one_h[k] * BOHR_IN_ANGSTROM +
                   charge * center_of_mass_forward_one_h[k]) -
             c1 * (dipole_backward_one_h[k] * BOHR_IN_ANGSTROM +
                   charge * center_of_mass_backward_one_h[k]) -
             c0 * (dipole_backward_two_h[k] * BOHR_IN_ANGSTROM +
                   charge * center_of_mass_backward_two_h[k])) *
            one_over_displacement;
        //cpu_dipole_jacobian_[i * N_components + j * values_per_atom + k] =
        //    (c0 * (dipole_forward_two_h[k] * BOHR_IN_ANGSTROM) +
        //     c1 * (dipole_forward_one_h[k] * BOHR_IN_ANGSTROM) -
        //     c1 * (dipole_backward_one_h[k] * BOHR_IN_ANGSTROM)-
        //     c0 * (dipole_backward_two_h[k] * BOHR_IN_ANGSTROM)) *
        //    one_over_displacement;
      }
      // Restore positions
      displace_atom<<<1, 1>>>(
          index, 
          atom_copy.position_per_atom.data(), 
          2*displacement); // -2h + 2h = 0
      CUDA_CHECK_KERNEL
      center_of_mass_forward_one_h[i] = old_center_of_mass;
      center_of_mass_forward_two_h[i] = old_center_of_mass;
      center_of_mass_backward_one_h[i] = old_center_of_mass;
      center_of_mass_backward_two_h[i] = old_center_of_mass;
    }
  }
}


void Cavity::cavity_force(const int step) {
  // TODO
}

void Cavity::write_dipole(const int step)
{
  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(file_, "%d%20.10e%20.10e%20.10e", step, cpu_dipole_[0], cpu_dipole_[1], cpu_dipole_[2]);
  for (int i = 0; i < cpu_dipole_jacobian_.size(); i++) {
    fprintf(file_, "%20.10e", cpu_dipole_jacobian_[i]);
  }
  fprintf(file_, "\n");
  fflush(file_);
}

void Cavity::postprocess()
{
  if (enabled_) {
    fclose(file_);
    enabled_ = false;
  }
}
