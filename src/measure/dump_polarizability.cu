/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <vector>
#include <cstring>

static __global__ void sum_polarizability(
  const int N, const int number_of_patches, const double* g_virial_per_atom, double* g_pol)
{
  //<<<6, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ double s_p[1024];
  double p = 0.0;

  // Data is in the order x1,.... xN, y1,...yN etc.
  // Each block sums each component, block 0 is x etc.
  // need to translate blockIdx to component index
  // in g_virial_per_atom
  const int blockToCompIdx[6] = {0, 1, 2, 3, 5, 7};
  const unsigned int componentIdx = blockToCompIdx[blockIdx.x] * N;

  // 1024 threads, each summing a patch of N/1024 atoms
  for (int patch = 0; patch < number_of_patches; ++patch) {
    int atomIdx = tid + patch * 1024;
    if (atomIdx < N)
      p += g_virial_per_atom[componentIdx + atomIdx];
  }

  // save the sum for this patch
  s_p[tid] = p;
  __syncthreads();

  // aggregate the patches in parallel

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_p[tid] += s_p[tid + offset];
    }
    __syncthreads();
  }

  // save the final value
  if (tid == 0) {
    g_pol[bid] = s_p[0];
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
  }
  if (n1 == 0) {
    // Only need to set g_pol to zero once
    g_pol[0] = 0.0;
    g_pol[1] = 0.0;
    g_pol[2] = 0.0;
    g_pol[3] = 0.0;
    g_pol[4] = 0.0;
    g_pol[5] = 0.0;
  }
}

Dump_Polarizability::Dump_Polarizability(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "dump_polarizability";
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

void Dump_Polarizability::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  if (dump_) {
    std::string filename_ = "polarizability.out";
    file_ = my_fopen(filename_.c_str(), "a");
    gpu_pol_.resize(6);
    cpu_pol_.resize(6);

    // Set up a local copy of the Atoms, on which to compute the dipole
    // Typically in GPUMD we are limited by computational speed, not memory,
    // so we can sacrifice a bit of memory to skip having to recompute the forces
    // & virials with the original potential
    atom_copy.number_of_atoms = atom.number_of_atoms;
    atom_copy.force_per_atom.resize(atom.number_of_atoms * 3);
    atom_copy.virial_per_atom.resize(atom.number_of_atoms * 9);
    atom_copy.potential_per_atom.resize(atom.number_of_atoms);
    // make sure that the second potential is actually a polarizability model.
    if (force.potentials.size() != 2) {
      PRINT_INPUT_ERROR("dump_polarizability requires two potentials to be specified.");
    }
    // Multiple potentials may only be used with NEPs, so we know that
    // the second potential must be an NEP
    if (force.potentials[1]->nep_model_type != 2) {
      PRINT_INPUT_ERROR(
        "dump_polarizability requires the second NEP potential to be a dipole model.");
    }
  }
}

void Dump_Polarizability::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
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
    gpu_pol_.data());
  GPU_CHECK_KERNEL

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
  const int number_of_threads = 1024;
  const int number_of_atoms_per_thread = (number_of_atoms - 1) / number_of_threads + 1;
  sum_polarizability<<<6, number_of_threads>>>(
    number_of_atoms, number_of_atoms_per_thread, atom_copy.virial_per_atom.data(), gpu_pol_.data());
  GPU_CHECK_KERNEL

  // Transfer gpu_sum to the CPU
  gpu_pol_.copy_to_host(cpu_pol_.data());
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

void Dump_Polarizability::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (dump_) {
    fclose(file_);
    dump_ = false;
  }
}
