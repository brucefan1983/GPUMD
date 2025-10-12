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
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Calculate the time derivative of the polarization of the system and output to dpdt.out
--------------------------------------------------------------------------------------------------*/

#include "compute_dpdt.cuh"
#include "force/force.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <vector>

namespace{

void __global__ gpu_compute_dpdt(
  const int num_atoms,
  const float* g_bec,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  float* g_dpdt_x,
  float* g_dpdt_y,
  float* g_dpdt_z)
{
  const int atom_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom_id < num_atoms) {
    float bec[9] = {0.0f};
    for (int d = 0; d < 9; ++d) {
      bec[d] = g_bec[atom_id + d * num_atoms];
    }
    const float vx = g_vx[atom_id];
    const float vy = g_vy[atom_id];
    const float vz = g_vz[atom_id];
    g_dpdt_x[atom_id] = bec[0] * vx + bec[1] * vy + bec[2] * vz;
    g_dpdt_y[atom_id] = bec[3] * vx + bec[4] * vy + bec[5] * vz;
    g_dpdt_z[atom_id] = bec[6] * vx + bec[7] * vy + bec[8] * vz;
  }
}

void __global__ gpu_sum_dpdt(const int N, const float* g_dpdt_per_atom, float* g_dpdt)
{
  const int tid = threadIdx.x;
  const int number_of_batches = (N - 1) / 1024 + 1;

  __shared__ float s_data[1024];
  s_data[tid] = 0.0f;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      s_data[tid] += g_dpdt_per_atom[n + N * blockIdx.x];
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_dpdt[blockIdx.x] = s_data[0];
  }
}

}

void Compute_dpdt::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  fid = fopen("dpdt.out", "a");
  gpu_dpdt_per_atom.resize(3 * atom.number_of_atoms);
  gpu_dpdt_total.resize(3);
  cpu_dpdt_total.resize(3);
  p_integral[0] = 0.0;
  p_integral[1] = 0.0;
  p_integral[2] = 0.0;
  p_integral_time = 0.0;
  p_integral_dt = time_step * sample_interval;
}

void Compute_dpdt::process(
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
  if ((step + 1) % sample_interval != 0)
    return;

  const int N = atom.number_of_atoms;
  GPU_Vector<float>& bec = force.potentials[0]->get_bec_reference();
  gpu_compute_dpdt<<<(N - 1) / 64 + 1, 64>>>(
    N,
    bec.data(),
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + N,
    atom.velocity_per_atom.data() + N * 2,
    gpu_dpdt_per_atom.data(),
    gpu_dpdt_per_atom.data() + N,
    gpu_dpdt_per_atom.data() + N * 2);
  GPU_CHECK_KERNEL

  gpu_sum_dpdt<<<3, 1024>>>(N, gpu_dpdt_per_atom.data(), gpu_dpdt_total.data());
  GPU_CHECK_KERNEL

  p_integral_time += p_integral_dt * TIME_UNIT_CONVERSION;
  gpu_dpdt_total.copy_to_host(cpu_dpdt_total.data());
  for (int d = 0; d < 3; ++d) {
    p_integral[d] += cpu_dpdt_total[d] * p_integral_dt; //  e A
    cpu_dpdt_total[d] /= TIME_UNIT_CONVERSION; // e A / fs
  }
  fprintf(fid, "%g %g %g %g %g %g %g\n", 
    p_integral_time,
    cpu_dpdt_total[0], 
    cpu_dpdt_total[1], 
    cpu_dpdt_total[2], 
    p_integral[0], 
    p_integral[1], 
    p_integral[2]);
  fflush(fid);
}

void Compute_dpdt::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(fid);
}

void Compute_dpdt::parse(const char** param, int num_param)
{
  printf("Compute dp/dt.\n");

  if (!check_is_nep_charge()) {
    PRINT_INPUT_ERROR("cannot use compute_dpdt for a non-NEP-Charge model.\n");
  }

  if (num_param != 2) {
    PRINT_INPUT_ERROR("compute_dpdt should have 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for compute_dpdt should be an integer number.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("sample interval for compute_dpdt should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);
}

Compute_dpdt::Compute_dpdt(const char** param, int num_param)
{
  parse(param, num_param);
  property_name = "compute_dpdt";
}
