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
The semi-grand canonical (SGC) ensemble for MCMD.
------------------------------------------------------------------------------*/

#include "mc_ensemble_sgc.cuh"

MC_Ensemble_SGC::MC_Ensemble_SGC(
  int num_steps_mc_input,
  bool is_vcsgc_input,
  std::vector<std::string>& species_input,
  std::vector<int>& types_input,
  std::vector<double>& mu_or_phi_input,
  double kappa_input)
{
  num_steps_mc = num_steps_mc_input;
  is_vcsgc = is_vcsgc_input;
  species = species_input;
  types = types_input;
  mu_or_phi = mu_or_phi_input;
  kappa = kappa_input;
  NN_ij.resize(1);
  NL_ij.resize(1000);
}

MC_Ensemble_SGC::~MC_Ensemble_SGC(void) { mc_output.close(); }

static __global__ void get_types(
  const int N,
  const int i,
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
  }
}

static __global__ void find_local_types(
  const int N_local,
  const int* atom_local,
  const int* g_type_before,
  const int* g_type_after,
  int* g_local_type_before,
  int* g_local_type_after)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < N_local) {
    int n = atom_local[k];
    g_local_type_before[k] = g_type_before[n];
    g_local_type_after[k] = g_type_after[n];
  }
}

static __global__ void get_neighbors_of_i(
  const int N,
  const Box box,
  const int i,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_i,
  int* g_NL_i)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double x0 = g_x[n];
    double y0 = g_y[n];
    double z0 = g_z[n];
    double x0i = g_x[i] - x0;
    double y0i = g_y[i] - y0;
    double z0i = g_z[i] - z0;

    apply_mic(box, x0i, y0i, z0i);
    float distance_square_i = float(x0i * x0i + y0i * y0i + z0i * z0i);

    if (distance_square_i < rc_radial_square) {
      g_NL_i[atomicAdd(g_NN_i, 1)] = n;
    }
  }
}

static __global__ void create_inputs_for_energy_calculator(
  const int N,
  const int N_local,
  const int* atom_local,
  const Box box,
  const float rc_radial_square,
  const float rc_angular_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* g_type_before,
  const int* g_type_after,
  int* g_NN_radial,
  int* g_NN_angular,
  int* g_t2_radial_before,
  int* g_t2_radial_after,
  int* g_t2_angular_before,
  int* g_t2_angular_after,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n2 < N) {
    double x2 = g_x[n2];
    double y2 = g_y[n2];
    double z2 = g_z[n2];

    for (int k = 0; k < N_local; ++k) {
      int n1 = atom_local[k];
      if (n1 == n2) {
        continue;
      }
      double x12 = x2 - g_x[n1];
      double y12 = y2 - g_y[n1];
      double z12 = z2 - g_z[n1];
      apply_mic(box, x12, y12, z12);
      float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
      if (distance_square < rc_radial_square) {
        int count_radial = atomicAdd(&g_NN_radial[k], 1);
        int index_radial = count_radial * N_local + k;
        g_t2_radial_before[index_radial] = g_type_before[n2];
        g_t2_radial_after[index_radial] = g_type_after[n2];
        g_x12_radial[index_radial] = float(x12);
        g_y12_radial[index_radial] = float(y12);
        g_z12_radial[index_radial] = float(z12);
      }
      if (distance_square < rc_angular_square) {
        int count_angular = atomicAdd(&g_NN_angular[k], 1);
        int index_angular = count_angular * N_local + k;
        g_t2_angular_before[index_angular] = g_type_before[n2];
        g_t2_angular_after[index_angular] = g_type_after[n2];
        g_x12_angular[index_angular] = float(x12);
        g_y12_angular[index_angular] = float(y12);
        g_z12_angular[index_angular] = float(z12);
      }
    }
  }
}

// a kernel with a single thread <<<1, 1>>>
static __global__ void gpu_flip(
  const int i,
  const int type_j,
  int* g_type,
  double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  g_type[i] = type_j;
  g_mass[i] = 0; // TODO
  g_vx[i] = 0;   // TODO
  g_vy[i] = 0;   // TODO
  g_vz[i] = 0;   // TODO
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

static bool allowed_species(std::string& species_found, std::vector<std::string>& species)
{
  for (auto s : species) {
    if (s == species_found) {
      return true;
    }
  }
  return false;
}

void MC_Ensemble_SGC::compute(
  int md_step,
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& groups,
  int grouping_method,
  int group_id)
{
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {
    printf("Cannot use small box for MCMD.\n");
    exit(1);
  }

  if (type_before.size() < atom.number_of_atoms) {
    type_before.resize(atom.number_of_atoms);
    type_after.resize(atom.number_of_atoms);
  }

  int group_size =
    grouping_method >= 0 ? groups[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  std::uniform_int_distribution<int> r1(0, group_size - 1);

  int num_accepted = 0;
  for (int step = 0; step < num_steps_mc; ++step) {
    int i = -1;
    int type_i = -1;
    std::string species_found;
    while (!allowed_species(species_found, species)) {
      i = grouping_method >= 0
            ? groups[grouping_method]
                .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
            : r1(rng);
      std::cout << "i=" << i << std::endl;
      species_found = atom.cpu_atom_symbol[i];
      type_i = atom.cpu_type[i];
      std::cout << "atom.cpu_atom_symbol[i]=" << atom.cpu_atom_symbol[i] << std::endl;
      std::cout << "type_i = " << type_i << std::endl;
    }

    int type_j = type_i;
    std::string species_new;
    std::uniform_int_distribution<int> rand_int2(0, types.size() - 1);
    while (type_j == type_i) {
      int random_index = rand_int2(rng);
      type_j = types[random_index];
      species_new = species[random_index];
      std::cout << "species_new = " << species_new << std::endl;
      std::cout << "type_j = " << type_j << std::endl;
    }

    exit(1);

    CHECK(cudaMemset(NN_ij.data(), 0, sizeof(int)));
    get_neighbors_of_i<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data());
    CUDA_CHECK_KERNEL

    int NN_ij_cpu;
    NN_ij.copy_to_host(&NN_ij_cpu);

    get_types<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms, i, type_j, atom.type.data(), type_before.data(), type_after.data());
    CUDA_CHECK_KERNEL

    find_local_types<<<(NN_ij_cpu - 1) / 64 + 1, 64>>>(
      NN_ij_cpu,
      NL_ij.data(),
      type_before.data(),
      type_after.data(),
      local_type_before.data(),
      local_type_after.data());
    CUDA_CHECK_KERNEL

    CHECK(cudaMemset(NN_radial.data(), 0, sizeof(int) * NN_radial.size()));
    CHECK(cudaMemset(NN_angular.data(), 0, sizeof(int) * NN_angular.size()));
    create_inputs_for_energy_calculator<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      NN_ij_cpu,
      NL_ij.data(),
      box,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      nep_energy.paramb.rc_angular * nep_energy.paramb.rc_angular,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      type_before.data(),
      type_after.data(),
      NN_radial.data(),
      NN_angular.data(),
      t2_radial_before.data(),
      t2_radial_after.data(),
      t2_angular_before.data(),
      t2_angular_after.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data());
    CUDA_CHECK_KERNEL

    nep_energy.find_energy(
      NN_ij_cpu,
      NN_radial.data(),
      NN_angular.data(),
      local_type_before.data(),
      t2_radial_before.data(),
      t2_angular_before.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_before.data());

    nep_energy.find_energy(
      NN_ij_cpu,
      NN_radial.data(),
      NN_angular.data(),
      local_type_after.data(),
      t2_radial_after.data(),
      t2_angular_after.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      x12_angular.data(),
      y12_angular.data(),
      z12_angular.data(),
      pe_after.data());

    std::vector<float> pe_before_cpu(NN_ij_cpu);
    std::vector<float> pe_after_cpu(NN_ij_cpu);
    pe_before.copy_to_host(pe_before_cpu.data(), NN_ij_cpu);
    pe_after.copy_to_host(pe_after_cpu.data(), NN_ij_cpu);
    float pe_before_total = 0.0f;
    float pe_after_total = 0.0f;
    for (int n = 0; n < NN_ij_cpu; ++n) {
      pe_before_total += pe_before_cpu[n];
      pe_after_total += pe_after_cpu[n];
    }
    // printf("        per-atom energy before swapping = %g eV.\n", pe_before_total / NN_ij_cpu);
    // printf("        per-atom energy after swapping = %g eV.\n", pe_after_total / NN_ij_cpu);
    float energy_difference = pe_after_total - pe_before_total;
    std::uniform_real_distribution<float> r2(0, 1);
    float random_number = r2(rng);
    float probability = exp(-energy_difference / (K_B * temperature));

    if (random_number < probability) {
      ++num_accepted;

      atom.cpu_type[i] = type_j;
      atom.cpu_atom_symbol[i] = species_new;
      atom.cpu_mass[i] = 0; // TODO

      gpu_flip<<<1, 1>>>(
        i,
        type_j,
        atom.type.data(),
        atom.mass.data(),
        atom.velocity_per_atom.data(),
        atom.velocity_per_atom.data() + atom.number_of_atoms,
        atom.velocity_per_atom.data() + atom.number_of_atoms * 2);
    }
  }

  mc_output << md_step << "  " << num_accepted / double(num_steps_mc) << std::endl;
}
