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

#include "dump_dipole.cuh"
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

static __global__ void sum_diagonal(int N, const double* g_virial_per_atom, double* g_virial_sum)
{
  // TODO benchmark atomics, consider reductions
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    atomicAdd(&g_virial_sum[0], g_virial_per_atom[n1 + 0 * N]);
    atomicAdd(&g_virial_sum[1], g_virial_per_atom[n1 + 1 * N]);
    atomicAdd(&g_virial_sum[2], g_virial_per_atom[n1 + 2 * N]);
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
  g_virial_sum[0] = 0.0;
  g_virial_sum[1] = 0.0;
  g_virial_sum[2] = 0.0;
}

void Dump_Dipole::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump dipole\n");

  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_dipole should have 1 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  printf("   every %d steps.\n", dump_interval_);
}

void Dump_Dipole::preprocess(
  const int number_of_atoms, const int number_of_potentials, Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  force.set_multiple_potentials_mode("observe");
  if (dump_) {
    std::string filename_ = "dipole.out";
    file_ = my_fopen(filename_.c_str(), "a");
    gpu_dipole_.resize(3);
    cpu_dipole_.resize(3);

    // Set up a local copy of the Atoms, on which to compute the dipole
    // Typically in GPUMD we are limited by computational speed, not memory,
    // so we can sacrifice a bit of memory to skip having to recompute the forces
    // & virials with the original potential
    atom_copy.number_of_atoms = number_of_atoms;
    atom_copy.force_per_atom.resize(number_of_atoms * 3);
    atom_copy.virial_per_atom.resize(number_of_atoms * 9);
    atom_copy.potential_per_atom.resize(number_of_atoms);

    // make sure that the second potential is actually a dipole model.
    if (number_of_potentials != 2) {
      PRINT_INPUT_ERROR("dump_dipole requires two potentials to be specified.");
    }
    // Multiple potentials may only be used with NEPs, so we know that
    // the second potential must be an NEP
    if (force.potentials[1]->nep_model_type != 1) {
      PRINT_INPUT_ERROR("dump_dipole requires the second NEP potential to be a dipole model.");
    }
  }
}

void Dump_Dipole::process(
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
    gpu_dipole_.data());
  CUDA_CHECK_KERNEL

  // Compute the dipole
  // Use the positions and types from the existing atoms object,
  // but store the results in the local copy.
  force.potentials[1]->compute(
    box,
    atom.type,
    atom.position_per_atom,
    atom_copy.potential_per_atom,
    atom_copy.force_per_atom,
    atom_copy.virial_per_atom);

  // Aggregate virial_per_atom into dipole
  sum_diagonal<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, atom_copy.virial_per_atom.data(), gpu_dipole_.data());
  CUDA_CHECK_KERNEL

  // Transfer gpu_sum to the CPU
  gpu_dipole_.copy_to_host(cpu_dipole_.data());
  // Write properties
  write_dipole(step);
}

void Dump_Dipole::write_dipole(const int step)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(file_, "%d%20.10e%20.10e%20.10e\n", step, cpu_dipole_[0], cpu_dipole_[1], cpu_dipole_[2]);
  fflush(file_);
}

void Dump_Dipole::postprocess()
{
  if (dump_) {
    fclose(file_);
    dump_ = false;
  }
}
