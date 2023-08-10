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
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
The canonical ensemble for MCMD.
------------------------------------------------------------------------------*/

#include "mc_ensemble_canonical.cuh"

MC_Ensemble_Canonical::MC_Ensemble_Canonical(int num_steps_mc_input, double temperature_input)
{
  num_steps_mc = num_steps_mc_input;
  temperature = temperature_input;
}

MC_Ensemble_Canonical::~MC_Ensemble_Canonical(void)
{
  // nothing now
}

static __global__ void get_types(
  const int N,
  const int i,
  const int j,
  const int type_i,
  const int type_j,
  const int* g_type,
  int* g_type_before,
  int* g_type_after)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    g_type_before[n] = g_type[n];
    g_type_after[n] = g_type[n];
    if (n == i) {
      g_type_after[i] = type_j;
    }
    if (n == j) {
      g_type_after[j] = type_i;
    }
  }
}

static __global__ void create_inputs_for_energy_calculator(
  const int N,
  const Box box,
  const float rc_radial_square,
  const float rc_angular_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = 0; n2 < N; ++n2) {
      if (n2 == n1) {
        continue;
      }
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
      if (distance_square < rc_radial_square) {
        g_NL_radial[count_radial * N + n1] = n2;
        g_x12_radial[count_radial * N + n1] = float(x12);
        g_y12_radial[count_radial * N + n1] = float(y12);
        g_z12_radial[count_radial * N + n1] = float(z12);
        count_radial++;
      }
      if (distance_square < rc_angular_square) {
        g_NL_angular[count_angular * N + n1] = n2;
        g_x12_angular[count_angular * N + n1] = float(x12);
        g_y12_angular[count_angular * N + n1] = float(y12);
        g_z12_angular[count_angular * N + n1] = float(z12);
        count_angular++;
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

static int get_type(const int i, const int* g_type)
{
  int type_i = 0;
  CHECK(cudaMemcpy(&type_i, &g_type[i], sizeof(int), cudaMemcpyDeviceToHost));
  return type_i;
}

void MC_Ensemble_Canonical::compute(Atom& atom, Box& box)
{
  std::uniform_int_distribution<int> r1(0, atom.number_of_atoms - 1);

  for (int step = 0; step < num_steps_mc; ++step) {
    printf("    MC step %d, temperature = %g K.\n", step, temperature);

    int i = r1(rng);
    int type_i = get_type(i, atom.type.data());
    printf("        get atom %d with type %d.\n", i, type_i);
    int j = 0, type_j = type_i;
    while (type_i == type_j) {
      j = r1(rng);
      type_j = get_type(j, atom.type.data());
      printf("        get atom %d with type %d.\n", j, type_j);
    }
    printf(
      "        try to exchange atom %d with type %d and atom %d with type %d.\n",
      i,
      type_i,
      j,
      type_j);

    get_types<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      i,
      j,
      type_i,
      type_j,
      atom.type.data(),
      type_before.data(),
      type_after.data());
    CUDA_CHECK_KERNEL

    create_inputs_for_energy_calculator<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      nep_energy.paramb.rc_angular * nep_energy.paramb.rc_angular,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_radial.data(),
      NL_radial.data(),
      NN_angular.data(),
      NL_angular.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data());
    CUDA_CHECK_KERNEL

    std::vector<int> NN_radial_cpu(atom.number_of_atoms);
    std::vector<int> NN_angular_cpu(atom.number_of_atoms);
    NN_radial.copy_to_host(NN_radial_cpu.data(), atom.number_of_atoms);
    NN_angular.copy_to_host(NN_angular_cpu.data(), atom.number_of_atoms);
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      printf("%d %d\n", NN_radial_cpu[n], NN_angular_cpu[n]);
    }

    exit(1);
  }
}
