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
  NN_i.resize(1);
  NN_j.resize(1);
  NN_ij.resize(1);
  NL_i.resize(1000);
  NL_j.resize(1000);
  NL_ij.resize(1000);
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

static __global__ void get_neighbors_of_i_and_j(
  const int N,
  const Box box,
  const int i,
  const int j,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_i,
  int* g_NN_j,
  int* g_NL_i,
  int* g_NL_j)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double x0 = g_x[n];
    double y0 = g_y[n];
    double z0 = g_z[n];
    double x0i = g_x[i] - x0;
    double y0i = g_y[i] - y0;
    double z0i = g_z[i] - z0;
    double x0j = g_x[j] - x0;
    double y0j = g_y[j] - y0;
    double z0j = g_z[j] - z0;

    apply_mic(box, x0i, y0i, z0i);
    float distance_square_i = float(x0i * x0i + y0i * y0i + z0i * z0i);
    apply_mic(box, x0j, y0j, z0j);
    float distance_square_j = float(x0j * x0j + y0j * y0j + z0j * z0j);

    if (distance_square_i < rc_radial_square) {
      g_NL_i[atomicAdd(g_NN_i, 1)] = n;
    }
    if (distance_square_j < rc_radial_square) {
      g_NL_j[atomicAdd(g_NN_j, 1)] = n;
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

static bool check_if_small_box(const double rc, const Box& box)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.0 * rc) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.0 * rc) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.0 * rc) {
    is_small_box = true;
  }
  return is_small_box;
}

void MC_Ensemble_Canonical::compute(Atom& atom, Box& box)
{
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {
    printf("Cannot use small box for MCMD.\n");
    exit(1);
  }

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

    cudaMemset(NN_i.data(), 0, sizeof(int));
    cudaMemset(NN_j.data(), 0, sizeof(int));
    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      j,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_i.data(),
      NN_j.data(),
      NL_i.data(),
      NL_j.data());
    CUDA_CHECK_KERNEL

    // copy to host
    int NN_i_cpu, NN_j_cpu;
    int NL_i_cpu[1000], NL_j_cpu[1000];
    NN_i.copy_to_host(&NN_i_cpu);
    NN_j.copy_to_host(&NN_j_cpu);
    NL_i.copy_to_host(NL_i_cpu, NN_i_cpu);
    NL_j.copy_to_host(NL_j_cpu, NN_j_cpu);

    printf("        atom %d has %d neighbors:\n", i, NN_i_cpu);
    for (int k = 0; k < NN_i_cpu; ++k) {
      printf(" %d", NL_i_cpu[k]);
    }
    printf("\n");
    printf("        atom %d has %d neighbors:\n", j, NN_j_cpu);
    for (int k = 0; k < NN_j_cpu; ++k) {
      printf(" %d", NL_j_cpu[k]);
    }
    printf("\n");

    // check in host
    int NN_ij_cpu = 0;
    int NL_ij_cpu[1000];
    for (; NN_ij_cpu < NN_i_cpu; ++NN_ij_cpu) {
      NL_ij_cpu[NN_ij_cpu] = NL_i_cpu[NN_ij_cpu];
    }

    for (int k = 0; k < NN_j_cpu; ++k) {
      bool is_repeating = false;
      for (int m = 0; m < NN_i_cpu; ++m) {
        if (NL_j_cpu[k] == NL_i_cpu[m]) {
          is_repeating = true;
          break;
        }
      }
      if (!is_repeating) {
        NL_ij_cpu[NN_ij_cpu++] = NL_j_cpu[k];
      }
    }

    printf("        i and j has %d neighbors in total:\n", NN_ij_cpu);
    for (int k = 0; k < NN_ij_cpu; ++k) {
      printf(" %d", NL_ij_cpu[k]);
    }
    printf("\n");

    // copy to device
    NN_ij.copy_from_host(&NN_ij_cpu);
    NL_ij.copy_from_host(NL_ij_cpu, NN_ij_cpu);

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

    nep_energy.find_energy(
      atom.number_of_atoms,
      NN_radial.data(),
      NL_radial.data(),
      NN_angular.data(),
      NL_angular.data(),
      type_before.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_before.data());

    nep_energy.find_energy(
      atom.number_of_atoms,
      NN_radial.data(),
      NL_radial.data(),
      NN_angular.data(),
      NL_angular.data(),
      type_after.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_after.data());

    std::vector<float> pe_before_cpu(atom.number_of_atoms);
    std::vector<float> pe_after_cpu(atom.number_of_atoms);
    pe_before.copy_to_host(pe_before_cpu.data(), atom.number_of_atoms);
    pe_after.copy_to_host(pe_after_cpu.data(), atom.number_of_atoms);
    float pe_before_total = 0.0f;
    float pe_after_total = 0.0f;
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      pe_before_total += pe_before_cpu[n];
      pe_after_total += pe_after_cpu[n];
    }
    printf("per-atom energy before swapping = %g eV.\n", pe_before_total / atom.number_of_atoms);
    printf("per-atom energy after swapping = %g eV.\n", pe_after_total / atom.number_of_atoms);
    float energy_difference = pe_after_total - pe_before_total;
    std::uniform_real_distribution<float> r2(0, 1);
    float random_number = r2(rng);
    printf("random number = %g.\n", random_number);
    float probability = exp(-energy_difference / (K_B * temperature));
    printf("probability = %g.\n", probability);

    if (random_number < probability) {
      printf("the MC trail is accepted.\n");
    } else {
      printf("the MC trail is rejected.\n");
    }
  }
}
