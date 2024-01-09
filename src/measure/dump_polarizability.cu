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

#include "dump_polarizability.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <vector>

static __global__ void sum_polarizability(
  int N, const double* g_potential_per_atom, const double* g_virial_per_atom, double* g_pol)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  int x = n1 + 0 * N;
  int y = n1 + 1 * N;
  int z = n1 + 2 * N;
  int xy = n1 + 3 * N;
  int yz = n1 + 5 * N;
  int zx = n1 + 7 * N;

  if (n1 < N) {
    // Write the same polarizability values as in the NEP executable and NEP_CPU:
    // xx yy zz xy yz zx
    // atomicAdd(&g_pol[n1 + 0 * N], g_potential_per_atom[x] - g_virial_per_atom[x]); // xx
    // atomicAdd(&g_pol[n1 + 1 * N], g_potential_per_atom[y] - g_virial_per_atom[y]); // yy
    // atomicAdd(&g_pol[n1 + 2 * N], g_potential_per_atom[z] - g_virial_per_atom[z]); // zz
    atomicAdd(&g_pol[n1 + 0 * N], -g_virial_per_atom[x]);  // xx
    atomicAdd(&g_pol[n1 + 1 * N], -g_virial_per_atom[y]);  // yy
    atomicAdd(&g_pol[n1 + 2 * N], -g_virial_per_atom[z]);  // zz
    atomicAdd(&g_pol[n1 + 3 * N], -g_virial_per_atom[xy]); // xy
    atomicAdd(&g_pol[n1 + 4 * N], -g_virial_per_atom[yz]); // yz
    atomicAdd(&g_pol[n1 + 5 * N], -g_virial_per_atom[zx]); // zx
  }
}

static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial, double* g_pol)
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
    g_pol[n1 + 0 * N] = 0.0;
    g_pol[n1 + 1 * N] = 0.0;
    g_pol[n1 + 2 * N] = 0.0;
    g_pol[n1 + 3 * N] = 0.0;
    g_pol[n1 + 4 * N] = 0.0;
    g_pol[n1 + 5 * N] = 0.0;
  }
}

void Dump_Polarizability::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump polarizability\n");

  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_dipole should have 1 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  printf("   every %d steps.\n", dump_interval_);
}

void Dump_Polarizability::preprocess(const int number_of_atoms, Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  force.set_multiple_potentials_mode("observe");
  if (dump_) {
    std::string filename_ = "polarizability.out";
    file_ = my_fopen(filename_.c_str(), "a");
    gpu_pol_per_atom_.resize(number_of_atoms * 6);
    cpu_pol_per_atom_.resize(number_of_atoms * 6);
    cpu_pol_.resize(6);

    // Set up a local copy of the Atoms, on which to compute the dipole
    // Typically in GPUMD we are limited by computational speed, not memory,
    // so we can sacrifice a bit of memory to skip having to recompute the forces
    // & virials with the original potential
    atom_copy.number_of_atoms = number_of_atoms;
    atom_copy.force_per_atom.resize(number_of_atoms * 3);
    atom_copy.virial_per_atom.resize(number_of_atoms * 9);
    atom_copy.potential_per_atom.resize(number_of_atoms);
  }
}

void Dump_Polarizability::process(
  int step,
  const double global_time,
  const int number_of_atoms_fixed,
  std::vector<Group>& group,
  Box& box,
  Atom& atom,
  Force& force)
{
  // Only run if should dump, since forces have to be recomputed with each potential.
  if (!dump_)
    return;
  if (((step + 1) % dump_interval_ != 0))
    return;
  const int number_of_atoms = atom_copy.number_of_atoms;

  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom_copy.force_per_atom.data(),
    atom_copy.force_per_atom.data() + number_of_atoms,
    atom_copy.force_per_atom.data() + number_of_atoms * 2,
    atom_copy.potential_per_atom.data(),
    atom_copy.virial_per_atom.data(),
    gpu_pol_per_atom_.data());
  CUDA_CHECK_KERNEL

  // Compute the dipole
  // Use the positions and types from the existing atoms object,
  // but store the results in the local copy.
  // TODO make sure that the second potential is actually a dipole model.
  force.potentials[1]->compute(
    box,
    atom.type,
    atom.position_per_atom,
    atom_copy.potential_per_atom,
    atom_copy.force_per_atom,
    atom_copy.virial_per_atom);

  // Aggregate virial_per_atom into dipole
  sum_polarizability<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom_copy.potential_per_atom.data(),
    atom_copy.virial_per_atom.data(),
    gpu_pol_per_atom_.data());
  CUDA_CHECK_KERNEL

  // Transfer gpu_sum to the CPU
  gpu_pol_per_atom_.copy_to_host(cpu_pol_per_atom_.data());
  // Sum up per atom
  for (int i = 0; i < 6; i++) {
    cpu_pol_[i] = 0.0;
    for (int j = 0; j < number_of_atoms; j++) {
      cpu_pol_[i] += cpu_pol_per_atom_[j + i * number_of_atoms];
    }
  }
  // Write properties
  write_polarizability(step);
}

void Dump_Polarizability::write_polarizability(const int step)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  // Write the same polarizability values as the NEP executable:
  // xx, yy, zz, xy, yz, zx
  fprintf(
    file_,
    "%d%20.10e%20.10e%20.10e%20.10e%20.10e%20.10e\n",
    step,
    cpu_pol_[0],
    cpu_pol_[1],
    cpu_pol_[2],
    cpu_pol_[3],
    cpu_pol_[4],
    cpu_pol_[5]);
  fflush(file_);
}

void Dump_Polarizability::postprocess()
{
  if (dump_) {
    fclose(file_);
    dump_ = false;
  }
}
