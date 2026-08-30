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

Dump_Polarizability::Dump_Polarizability(const char** param, int num_param)
{
  parse(param, num_param);
  action_name = "dump_polarizability";
}

void Dump_Polarizability::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump polarizability\n");

  if (num_param != 3) {
    PRINT_INPUT_ERROR("dump_polarizability should have 2 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  file_potential_ = param[2];
  printf("   every %d steps.\n", dump_interval_);
  printf("   response potential: %s.\n", file_potential_.c_str());
}

void Dump_Polarizability::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (dump_) {
    std::string filename_ = "polarizability.out";
    file_ = my_fopen(filename_.c_str(), "a");
    fprintf(file_, "# dump_polarizability %d\n", dump_interval_);
    fprintf(file_, "# format_version 1\n");
    fprintf(file_, "# num_atoms %d\n", atom.number_of_atoms);
    fprintf(
      file_,
      "# cell %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
      box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
      box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
      box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
    fprintf(file_, "# dt_output %.10e fs\n", time_step * dump_interval_ * TIME_UNIT_CONVERSION);
    fprintf(file_, "# columns step pol_xx pol_yy pol_zz pol_xy pol_yz pol_zx\n");
    gpu_pol_.resize(6);
    cpu_pol_.resize(6);

    nep_response_.reset(new NEP_Response(file_potential_.c_str(), atom));
    if (!nep_response_->is_polarizability()) {
      PRINT_INPUT_ERROR("dump_polarizability requires a nep4_polarizability model.");
    }
  }
}

void Dump_Polarizability::end_of_step(
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
  // Only evaluate the response at requested output steps.
  if (!dump_)
    return;
  if (((step + 1) % dump_interval_ != 0))
    return;
  const int number_of_atoms = atom.number_of_atoms;
  const GPU_Vector<double>& response = nep_response_->compute(box, atom.position_per_atom);

  const int number_of_threads = 1024;
  const int number_of_atoms_per_thread = (number_of_atoms - 1) / number_of_threads + 1;
  sum_polarizability<<<6, number_of_threads>>>(
    number_of_atoms, number_of_atoms_per_thread, response.data(), gpu_pol_.data());
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

void Dump_Polarizability::post_run(
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
