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

/*----------------------------------------------------------------------------80
The canonical ensemble for MCMD.
------------------------------------------------------------------------------*/

#include "mc_ensemble_canonical.cuh"
#include "utilities/gpu_macro.cuh"

MC_Ensemble_Canonical::MC_Ensemble_Canonical(
  const char** param, int num_param, int num_steps_mc_input)
  : MC_Ensemble(param, num_param)
{
  num_steps_mc = num_steps_mc_input;
  NN_ij.resize(1);
  NL_ij.resize(1000);
}

MC_Ensemble_Canonical::~MC_Ensemble_Canonical(void) { mc_output.close(); }

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

static __global__ void get_neighbors_of_i_and_j(
  const int N,
  const Box box,
  const int i,
  const int j,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_ij,
  int* g_NL_ij)
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
      g_NL_ij[atomicAdd(g_NN_ij, 1)] = n;
    } else {
      if (distance_square_j < rc_radial_square) {
        g_NL_ij[atomicAdd(g_NN_ij, 1)] = n;
      }
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
static __global__ void exchange(
  const int i,
  const int j,
  const int type_i,
  const int type_j,
  int* g_type,
  double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  g_type[i] = type_j;
  g_type[j] = type_i;

  double mass_i = g_mass[i];
  g_mass[i] = g_mass[j];
  g_mass[j] = mass_i;

  double vx_i = g_vx[i];
  g_vx[i] = g_vx[j];
  g_vx[j] = vx_i;

  double vy_i = g_vy[i];
  g_vy[i] = g_vy[j];
  g_vy[j] = vy_i;

  double vz_i = g_vz[i];
  g_vz[i] = g_vz[j];
  g_vz[j] = vz_i;
}

void MC_Ensemble_Canonical::compute(
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

    int i = grouping_method >= 0
              ? groups[grouping_method]
                  .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
              : r1(rng);
    int type_i = atom.cpu_type[i];
    int j = 0, type_j = type_i;
    while (type_i == type_j) {
      j = grouping_method >= 0
            ? groups[grouping_method]
                .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
            : r1(rng);
      type_j = atom.cpu_type[j];
    }

    CHECK(gpuMemset(NN_ij.data(), 0, sizeof(int)));
    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      j,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data());
    GPU_CHECK_KERNEL

    int NN_ij_cpu;
    NN_ij.copy_to_host(&NN_ij_cpu);

    get_types<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      i,
      j,
      type_i,
      type_j,
      atom.type.data(),
      type_before.data(),
      type_after.data());
    GPU_CHECK_KERNEL

    find_local_types<<<(NN_ij_cpu - 1) / 64 + 1, 64>>>(
      NN_ij_cpu,
      NL_ij.data(),
      type_before.data(),
      type_after.data(),
      local_type_before.data(),
      local_type_after.data());
    GPU_CHECK_KERNEL

    CHECK(gpuMemset(NN_radial.data(), 0, sizeof(int) * NN_radial.size()));
    CHECK(gpuMemset(NN_angular.data(), 0, sizeof(int) * NN_angular.size()));
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
    GPU_CHECK_KERNEL

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
      atom.cpu_type[j] = type_i;

      auto atom_symbol_i = atom.cpu_atom_symbol[i];
      atom.cpu_atom_symbol[i] = atom.cpu_atom_symbol[j];
      atom.cpu_atom_symbol[j] = atom_symbol_i;

      double mass_i = atom.cpu_mass[i];
      atom.cpu_mass[i] = atom.cpu_mass[j];
      atom.cpu_mass[j] = mass_i;

      exchange<<<1, 1>>>(
        i,
        j,
        type_i,
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


static __global__ void get_shpere_atoms(
  const int N,
  const int local_N,
  const int* local_index,
  const Box box,
  const int i,
  const int j,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* local_sphere_flags)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < local_N) {
    int index = local_index[n];
    double x0 = g_x[index];
    double y0 = g_y[index];
    double z0 = g_z[index];
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
      local_sphere_flags[n] = 1;
      local_sphere_flags[n + local_N] = 1;
      local_sphere_flags[n + 2 * local_N] = 1;
    } else {
      if (distance_square_j < rc_radial_square) {
        local_sphere_flags[n] = 1;
        local_sphere_flags[n + local_N] = 1;
        local_sphere_flags[n + 2 * local_N] = 1;
      }
    }
  }
}

static __global__ void get_outer_atoms(
  const int N,
  const int local_N,
  const int* local_index,
  const Box box,
  const int i,
  const int j,
  const float rcmin_radial_square,
  const float rcmax_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* outer_atoms_flags)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < local_N) {
    int index = local_index[n];
    double x0 = g_x[index];
    double y0 = g_y[index];
    double z0 = g_z[index];
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

    if (distance_square_i < rcmax_radial_square && distance_square_i > rcmin_radial_square) {
      outer_atoms_flags[n] += 1;
    } else {
      if (distance_square_j < rcmax_radial_square && distance_square_j > rcmin_radial_square) {
        outer_atoms_flags[n] += 1;
      }
    }
  }
}


//copy from a to b
template <typename T>
static __global__ void copy
(
  T* a,
  T* b,
  int* local_index,
  int global_N,
  int local_N,
  int dimension)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < local_N)
  {
    int index = local_index[n];
    for (int i = 0; i < dimension; i++)
    {
      b[n + i * local_N] = a[index + i * global_N];
    }
  }
}


//copy from b to a
template <typename T>
static __global__ void copy_back
(
  T* a,
  T* b,
  int* local_index,
  int global_N,
  int local_N,
  int dimension)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < local_N)
  {
    int index = local_index[n];
    for (int i = 0; i < dimension; i++)
    {
      a[index + i * global_N] = b[n + i * local_N];
    }
  }
}


static __global__ void gpu_sum(const int size, double* a, double* result)
{
  int number_of_patches = (size - 1) / 1024 + 1;
  int tid = threadIdx.x;
  int n, patch;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * 1024;
    if (n < size)
      data[tid] += a[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      data[tid] += data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = data[0];
}

static __global__ void get_displacement(
  const Box box,
  double* global_position,
  double* local_position,
  int global_N,
  int local_N,
  int* local_index,
  double* outer_atoms_flags,
  double* displacement)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < local_N)
  {
    int index = local_index[n];
    double displace = 0;
    double r12[3];
    for (int i = 0; i < 3; i++)
    {
      r12[i] = (global_position[index + i * global_N] - local_position[n + i * local_N]);
    }
    apply_mic(box, r12[0], r12[1], r12[2]);
    displace = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
    displacement[n] = sqrt(displace) * outer_atoms_flags[n];
  }
}


__global__ void gpu_calculate_max(
  const int size,
  const int number_of_rounds,
  const double* array,
  double* max)
{
  const int tid = threadIdx.x;

  __shared__ double s_max[1024]; // Shared memory for max values
  s_max[tid] = 0;         

  double max_value = 0;   // Initialize local max

  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = tid + round * 1024;
    if (n < size) {
      const double f = array[n];
      if (f > max_value)
        max_value = f;           // Update local max
    }
  }

  s_max[tid] = max_value;        // Write local max to shared memory
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid + offset] > s_max[tid]) {
        s_max[tid] = s_max[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    max[0] = s_max[0]; // Block's final max written to global memory
  }
}

double get_outer_average_displacement(
  const Box box,
  double* global_position,
  double* local_position,
  int global_N,
  int local_N,
  int* local_index,
  double* outer_atoms_flags,
  double* max)
{
  GPU_Vector<double> displacement(local_N);
  get_displacement<<<(local_N - 1) / 64 + 1, 64>>>(
    box,
    global_position,
    local_position,
    global_N,
    local_N,
    local_index,
    outer_atoms_flags,
    displacement.data());
  gpuDeviceSynchronize();
  
  //calculate the average displacement
  const int number_of_rounds = (local_N - 1) / 1024 + 1;
  gpu_calculate_max<<<1, 1024>>>(local_N, number_of_rounds, displacement.data(), max);

  GPU_Vector<double> result(1);
  gpu_sum<<<1, 1024>>>(local_N, displacement.data(), result.data());
  GPU_Vector<double> outer_number(1);
  gpu_sum<<<1, 1024>>>(local_N, outer_atoms_flags, outer_number.data());
  gpuDeviceSynchronize();
  double number;
  outer_number.copy_to_host(&number, 1);
  double ans;
  result.copy_to_host(&ans, 1);
  ans /= number;
  return ans;
}

void build_all_atoms(
  Atom& atom,
  Atom& local_atoms,
  int local_N,
  int* local_index)
{
  copy_back<<<(local_N - 1) / 64 + 1, 64>>>(
    atom.position_per_atom.data(),
    local_atoms.position_per_atom.data(),
    local_index,
    atom.number_of_atoms,
    local_N,
    3);
  gpuDeviceSynchronize();
}

void build_local_atoms(
  Atom& atom,
  Atom& local_atoms,
  int local_N,
  int* local_index)
{
  copy<<<(local_N - 1) / 64 + 1, 64>>>(
    atom.mass.data(),
    local_atoms.mass.data(),
    local_index,
    atom.number_of_atoms,
    local_N,
    1);
  copy<<<(local_N - 1) / 64 + 1, 64>>>(
    atom.type.data(),
    local_atoms.type.data(),
    local_index,
    atom.number_of_atoms,
    local_N,
    1);
  copy<<<(local_N - 1) / 64 + 1, 64>>>(
    atom.position_per_atom.data(),
    local_atoms.position_per_atom.data(),
    local_index,
    atom.number_of_atoms,
    local_N,
    3);
  gpuDeviceSynchronize();
}

//implement the local simple MC
void MC_Ensemble_Canonical::compute_local(
    double scale_factor,
    double temperature,
    Force& force,
    int max_relaxation_step,
    double force_tolerance,
    Atom& atom,
    Box& box,
    std::vector<Group>& groups,
    int grouping_method,
    int group_id)
{
  if (check_if_small_box(nep_energy.paramb.rc_radial, box)) {
    printf("Cannot use small box for simple MC.\n");
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

    int i = grouping_method >= 0
              ? groups[grouping_method]
                  .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
              : r1(rng);
    int type_i = atom.cpu_type[i];
    int j = 0, type_j = type_i;
    while (type_i == type_j) {
      j = grouping_method >= 0
            ? groups[grouping_method]
                .cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + r1(rng)]
            : r1(rng);
      type_j = atom.cpu_type[j];
    }
    static bool isFirstCall = true;
    
    if (isFirstCall)
    {
      mc_output << "Maximum displacement" << "  " << "Average displacement" << "  " <<  "Accept ratio" << std::endl;
      isFirstCall = false;
    }
    
    CHECK(gpuMemset(NN_ij.data(), 0, sizeof(int)));
    NL_ij.resize(atom.number_of_atoms);
    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      j,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial  * scale_factor * scale_factor,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data());
    GPU_CHECK_KERNEL

    int NN_ij_cpu;
    NN_ij.copy_to_host(&NN_ij_cpu);

    get_types<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      i,
      j,
      type_i,
      type_j,
      atom.type.data(),
      type_before.data(),
      type_after.data());
    GPU_CHECK_KERNEL

    GPU_Vector<int> local_type_before_temp(atom.number_of_atoms);
    GPU_Vector<int> local_type_after_temp(atom.number_of_atoms);
    GPU_Vector<int> t2_radial_before_temp(atom.number_of_atoms * 1000);
    GPU_Vector<int> t2_radial_after_temp(atom.number_of_atoms * 1000);
    GPU_Vector<int> t2_angular_before_temp(atom.number_of_atoms * 1000);
    GPU_Vector<int> t2_angular_after_temp(atom.number_of_atoms * 1000);
    GPU_Vector<float> x12_radial_temp(atom.number_of_atoms * 1000);
    GPU_Vector<float> y12_radial_temp(atom.number_of_atoms * 1000);
    GPU_Vector<float> z12_radial_temp(atom.number_of_atoms * 1000);
    GPU_Vector<float> x12_angular_temp(atom.number_of_atoms * 1000);
    GPU_Vector<float> y12_angular_temp(atom.number_of_atoms * 1000);
    GPU_Vector<float> z12_angular_temp(atom.number_of_atoms * 1000);
    GPU_Vector<double> pe_before_temp(atom.number_of_atoms);

    find_local_types<<<(NN_ij_cpu - 1) / 64 + 1, 64>>>(
      NN_ij_cpu,
      NL_ij.data(),
      type_before.data(),
      type_after.data(),
      local_type_before_temp.data(),
      local_type_after_temp.data());
    GPU_CHECK_KERNEL

    NN_radial.resize(atom.number_of_atoms);
    NN_angular.resize(atom.number_of_atoms);
    CHECK(gpuMemset(NN_radial.data(), 0, sizeof(int) * NN_radial.size()));
    CHECK(gpuMemset(NN_angular.data(), 0, sizeof(int) * NN_angular.size()));
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
      t2_radial_before_temp.data(),
      t2_radial_after_temp.data(),
      t2_angular_before_temp.data(),
      t2_angular_after_temp.data(),
      x12_radial_temp.data(),
      y12_radial_temp.data(),
      z12_radial_temp.data(),
      x12_angular_temp.data(),
      y12_angular_temp.data(),
      z12_angular_temp.data());
    GPU_CHECK_KERNEL

    //get the before energy
    nep_energy.find_energy(
      NN_ij_cpu,
      NN_radial.data(),
      NN_angular.data(),
      local_type_before_temp.data(),
      t2_radial_before_temp.data(),
      t2_angular_before_temp.data(),
      x12_radial_temp.data(),
      y12_radial_temp.data(),
      z12_radial_temp.data(),
      x12_angular_temp.data(),
      y12_angular_temp.data(),
      z12_angular_temp.data(),
      pe_before_temp.data());

    //calculate the after energy
    exchange<<<1, 1>>>(
      i,
      j,
      type_i,
      type_j,
      atom.type.data(),
      atom.mass.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + atom.number_of_atoms,
      atom.velocity_per_atom.data() + atom.number_of_atoms * 2);


    Atom local_atoms;
    //find local atoms
    GPU_Vector<int> local_N; //the length of local_index
    GPU_Vector<int> local_index; // an array contains the index of the local atoms
    GPU_Vector<double> local_sphere_flags; // an array labels the shell atom
    local_N.resize(1);
    local_index.resize(atom.number_of_atoms);
    CHECK(gpuMemset(local_N.data(), 0, sizeof(int)));
    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      j,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial * (scale_factor + 1.2) * (scale_factor + 1.2),
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      local_N.data(),
      local_index.data());
    GPU_CHECK_KERNEL

    int local_N_cpu;
    local_N.copy_to_host(&local_N_cpu);
    local_sphere_flags.resize(local_N_cpu * 3);
    local_sphere_flags.fill(0);

    //get sphere atoms
    get_shpere_atoms<<<(local_N_cpu - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      local_N_cpu,
      local_index.data(),
      box,
      i,
      j,
      nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial * scale_factor * scale_factor,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      local_sphere_flags.data());


    //build the local_atoms
    local_atoms.number_of_atoms = local_N_cpu;
    local_atoms.position_per_atom.resize(3 * local_N_cpu);
    local_atoms.force_per_atom.resize(3 * local_N_cpu);
    local_atoms.type.resize(local_N_cpu);
    local_atoms.velocity_per_atom.resize(3 * local_N_cpu);
    local_atoms.mass.resize(local_N_cpu);
    local_atoms.potential_per_atom.resize(local_N_cpu);
    local_atoms.virial_per_atom.resize(9 * local_N_cpu);

    //initialize
    local_atoms.force_per_atom.fill(0);
    local_atoms.potential_per_atom.fill(0);
    local_atoms.velocity_per_atom.fill(0);
    local_atoms.virial_per_atom.fill(0);
  
    build_local_atoms(
      atom,
      local_atoms,
      local_N_cpu,
      local_index.data());


    //need to modify here
    force.potentials[0]->N2 = local_N_cpu;
    Minimizer_FIRE Minimizer(
      local_N_cpu,
       max_relaxation_step,
      force_tolerance);

    Minimizer.compute_local(
      force,
      box,
      local_atoms.position_per_atom,
      local_atoms.type,
      groups,
      local_sphere_flags,
      local_atoms.potential_per_atom,
      local_atoms.force_per_atom,
      local_atoms.virial_per_atom);

    std::vector<double> pe_before_cpu(NN_ij_cpu);
    std::vector<double> pe_after_cpu(local_N_cpu);
    pe_before_temp.copy_to_host(pe_before_cpu.data(), NN_ij_cpu);
    local_atoms.potential_per_atom.copy_to_host(pe_after_cpu.data(), local_N_cpu);
    double pe_before_total = 0.0f;
    double pe_after_total = 0.0f;
    for (int n = 0; n < NN_ij_cpu; ++n) {
      pe_before_total += pe_before_cpu[n];
    }
    for (int n = 0; n < local_N_cpu; ++n) {
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
      atom.cpu_type[j] = type_i;

      auto atom_symbol_i = atom.cpu_atom_symbol[i];
      atom.cpu_atom_symbol[i] = atom.cpu_atom_symbol[j];
      atom.cpu_atom_symbol[j] = atom_symbol_i;

      double mass_i = atom.cpu_mass[i];
      atom.cpu_mass[i] = atom.cpu_mass[j];
      atom.cpu_mass[j] = mass_i;

      //get the output data
      GPU_Vector<double> outer_atoms_flags(local_N_cpu);
      outer_atoms_flags.fill(0);
      get_outer_atoms<<<(local_N_cpu - 1) / 64 + 1, 64>>>(
        atom.number_of_atoms,
        local_N_cpu,
        local_index.data(),
        box,
        i,
        j,
        nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial * (scale_factor - 1) * (scale_factor - 1),
        nep_energy.paramb.rc_radial * nep_energy.paramb.rc_radial * scale_factor * scale_factor,
        atom.position_per_atom.data(),
        atom.position_per_atom.data() + atom.number_of_atoms,
        atom.position_per_atom.data() + atom.number_of_atoms * 2,
        outer_atoms_flags.data());
      gpuDeviceSynchronize();

      GPU_Vector<double> max_displacement(1);
      double output = get_outer_average_displacement(
        box,
        atom.position_per_atom.data(),
        local_atoms.position_per_atom.data(),
        atom.number_of_atoms,
        local_atoms.number_of_atoms,
        local_index.data(),
        outer_atoms_flags.data(),
        max_displacement.data());
      double max_value;
      max_displacement.copy_to_host(&max_value);

      mc_output << max_value << "  " << output << "  " << num_accepted / (double(step) + 1) << std::endl;

      //copy the relaxed local structure to the global structure
      build_all_atoms(
        atom,
        local_atoms,
        local_N_cpu,
        local_index.data());
    }
    else
    {
      exchange<<<1, 1>>>(
        i,
        j,
        type_j,
        type_i,
        atom.type.data(),
        atom.mass.data(),
        atom.velocity_per_atom.data(),
        atom.velocity_per_atom.data() + atom.number_of_atoms,
        atom.velocity_per_atom.data() + atom.number_of_atoms * 2);
    }
    //need modification
    force.potentials[0]->N2 = atom.number_of_atoms;
  }
}
