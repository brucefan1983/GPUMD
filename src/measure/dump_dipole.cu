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
#include "model/box.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <vector>

static __global__ void gpu_sum(const int N, const double* g_data, double* g_data_sum)
{
  int number_of_rounds = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[threadIdx.x] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = threadIdx.x + round * 1024;
    if (n < N) {
      s_data[threadIdx.x] += g_data[n + blockIdx.x * N];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    g_data_sum[blockIdx.x] = s_data[0];
  }
}

static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
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
}

void Dump_Dipole::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump observer.\n");

  if (num_param != 3) {
    PRINT_INPUT_ERROR("dump_dipole should have 1 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval thermo should be an integer.");
  }
  printf("   every %d steps.\n", dump_interval_);
}

void Dump_Dipole::preprocess(const int number_of_atoms, Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  force.set_multiple_potentials_mode("observe");
  if (dump_) {
    std::string filename_ = "dipole.out";
    file_ = my_fopen(filename_.c_str(), "a");
    gpu_total_virial_.resize(6);
    cpu_total_virial_.resize(6);
    cpu_force_per_atom_.resize(number_of_atoms * 3);
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
  const int number_of_atoms = atom.type.size();
  initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom.force_per_atom.data(),
    atom.force_per_atom.data() + number_of_atoms,
    atom.force_per_atom.data() + number_of_atoms * 2,
    atom.potential_per_atom.data(),
    atom.virial_per_atom.data());
  CUDA_CHECK_KERNEL
  // Compute new potential properties
  // the dipoles are stored in the virials
  // only the diagonal terms matter
  // TODO make sure that the second potential is actually a dipole model.
  force.potentials[1]->compute(
    box,
    atom.type,
    atom.position_per_atom,
    atom.potential_per_atom,
    atom.force_per_atom,
    atom.virial_per_atom);

  // Aggregate force_per_atom into dipole
  // TODO use gpu_sum
  std::vector<double> dipole{0.0, 0.0, 0.0};
  // Write properties
  write_dipole(step, dipole);
}

void Dump_Dipole::write_dipole(const int step, std::vector<double>& dipole)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  // stress components are in Voigt notation: xx, yy, zz, yz, xz, xy
  fprintf(file_, "%d%20.10e%20.10e%20.10e", step, dipole[0], dipole[1], dipole[2]);
  fflush(file_);
}

void Dump_Dipole::postprocess()
{
  fclose(file_);
  dump_ = false;
}
