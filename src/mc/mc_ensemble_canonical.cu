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

    // std::vector<int> type_before_cpu(atom.number_of_atoms);
    // std::vector<int> type_after_cpu(atom.number_of_atoms);
    // type_before.copy_to_host(type_before_cpu.data(), atom.number_of_atoms);
    // type_after.copy_to_host(type_after_cpu.data(), atom.number_of_atoms);
    // for (int n = 0; n < atom.number_of_atoms; ++n) {
    // if (type_before_cpu[n] != type_after_cpu[n])
    // printf("%d\n", n);
    // }

    exit(1);
  }
}
