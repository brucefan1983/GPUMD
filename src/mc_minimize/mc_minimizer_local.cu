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
#pragma once

#include "mc_minimizer_local.cuh"

MC_Minimizer_Local::MC_Minimizer_Local(
  const char** param, int num_param,
    double scale_factor_input,
    double temperature_input,
    double force_tolerance_input,
    int max_relax_steps_input)
  : MC_Minimizer(param, num_param)
{
  scale_factor = scale_factor_input;
  temperature = temperature_input;
  force_tolerance = force_tolerance_input;
  max_relax_steps = max_relax_steps_input;
}

/*
this function can find out the atoms whose distance from two atom centers is within rc_radial
*/
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

// find the local index of ij
static __global__ void find_local_ij(
  const int i,
  const int j,
  const int local_N,
  int* local_index,
  int* local_i,
  int* local_j)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < local_N)
  {
    if (local_index[n] == i)
    {
        *local_i = n;
    }
    if (local_index[n] == j)
    {
        *local_j = n;
    }
  }
}

/*
this function is used for label the atoms within the rc_radial, which will return an array of [0, 1, 0, 0......].
1 stands for inside, and 0 stands for outside.
*/
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

/*
This function is used to mark atoms whose distance from two atom centers is between the range of rcmin and rcmax
*/
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

//copy from a to b, for position, dimension is 3, for mass, dimension is 1, and so on
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

//sum up
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

/*
calculate the displacement for the labelled atomsd
*/
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

//remove the energy of the background atoms
static __global__ void modify_energy(
  const int local_N,
  double* pe,
  double* local_sphere_flags)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < local_N)
  {
    pe[n] *= local_sphere_flags[n];
  } 
}

//calculate the max value
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

//copy the local structure to previous system
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
  copy_back<<<(local_N - 1) / 64 + 1, 64>>>(
    atom.type.data(),
    local_atoms.type.data(),
    local_index,
    atom.number_of_atoms,
    local_N,
    1);
  gpuDeviceSynchronize();
}

//construct a local structure
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
void MC_Minimizer_Local::compute(
    int trials,
    Force& force,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id)
{
  int group_size =
    grouping_method >= 0 ? group[grouping_method].cpu_size[group_id] : atom.number_of_atoms;
  std::uniform_int_distribution<int> r1(0, group_size - 1);

  int num_accepted = 0;
  for (int step = 0; step < trials; ++step) {

    int i = grouping_method >= 0
              ? group[grouping_method]
                  .cpu_contents[group[grouping_method].cpu_size_sum[group_id] + r1(rng)]
              : r1(rng);
    int type_i = atom.cpu_type[i];
    int j = 0, type_j = type_i;
    while (type_i == type_j) {
      j = grouping_method >= 0
            ? group[grouping_method]
                .cpu_contents[group[grouping_method].cpu_size_sum[group_id] + r1(rng)]
            : r1(rng);
      type_j = atom.cpu_type[j];
    }

    double rc_radius = force.potentials[0]->rc;
    Atom local_atoms;
    //find local atoms
    GPU_Vector<int> local_i;  //the local index of i atom
    GPU_Vector<int> local_j;  //the local index of j atom
    GPU_Vector<int> local_N; //the length of local_index
    GPU_Vector<int> local_index; // an array contains the index of the local atoms
    GPU_Vector<double> local_sphere_flags; // an array labels the shell atom
    local_N.resize(1);
    local_i.resize(1);
    local_j.resize(1);
    local_index.resize(atom.number_of_atoms);
    CHECK(gpuMemset(local_N.data(), 0, sizeof(int)));
    CHECK(gpuMemset(local_i.data(), 0, sizeof(int)));
    CHECK(gpuMemset(local_j.data(), 0, sizeof(int)));
    get_neighbors_of_i_and_j<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      box,
      i,
      j,
      rc_radius * rc_radius * (scale_factor + 1.2) * (scale_factor + 1.2),
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

    find_local_ij<<<(local_N_cpu - 1) / 64 + 1, 64>>>(
      i,
      j,
      local_N_cpu,
      local_index.data(),
      local_i.data(),
      local_j.data());
    int local_i_cpu;
    int local_j_cpu;
    local_i.copy_to_host(&local_i_cpu);
    local_j.copy_to_host(&local_j_cpu);

    //get sphere atoms
    get_shpere_atoms<<<(local_N_cpu - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      local_N_cpu,
      local_index.data(),
      box,
      i,
      j,
      rc_radius * rc_radius * scale_factor * scale_factor,
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
    
    //calculate the origin energy, WARNING: here the group is not correct but now this variable will not be used in compute()
    //need to modify here
    force.potentials[0]->N2 = local_N_cpu;
    force.compute(box,
    local_atoms.position_per_atom,
    local_atoms.type,
    group,
    local_atoms.potential_per_atom,
    local_atoms.force_per_atom,
    local_atoms.virial_per_atom);
    modify_energy<<<(local_N_cpu - 1) / 64 + 1, 64>>>(
      local_N_cpu,
      local_atoms.potential_per_atom.data(),
      local_sphere_flags.data());

    std::vector<double> pe_before_cpu(local_N_cpu);
    local_atoms.potential_per_atom.copy_to_host(pe_before_cpu.data(), local_N_cpu);

    exchange<<<1, 1>>>(
      local_i_cpu,
      local_j_cpu,
      type_i,
      type_j,
      local_atoms.type.data(),
      local_atoms.mass.data(),
      local_atoms.velocity_per_atom.data(),
      local_atoms.velocity_per_atom.data() + atom.number_of_atoms,
      local_atoms.velocity_per_atom.data() + atom.number_of_atoms * 2);

    Minimizer_FIRE Minimizer(
      local_N_cpu,
       max_relax_steps,
      force_tolerance);

    Minimizer.compute_label_atoms(
      force,
      box,
      local_atoms.position_per_atom,
      local_atoms.type,
      group,
      local_sphere_flags,
      local_atoms.potential_per_atom,
      local_atoms.force_per_atom,
      local_atoms.virial_per_atom);

    std::vector<double> pe_after_cpu(local_N_cpu);
    local_atoms.potential_per_atom.copy_to_host(pe_after_cpu.data(), local_N_cpu);
    double pe_before_total = 0;
    double pe_after_total = 0;
    for (int n = 0; n < local_N_cpu; ++n) {
      pe_before_total += pe_before_cpu[n];
    }
    for (int n = 0; n < local_N_cpu; ++n) {
      pe_after_total += pe_after_cpu[n];
    }
    //printf("        energy before swapping = %g.10 eV.\n", pe_before_total);
    //printf("        energy after swapping = %g.10 eV.\n", pe_after_total);
    double energy_difference = pe_after_total - pe_before_total;
    std::uniform_real_distribution<float> r2(0, 1);
    float random_number = r2(rng);
    double probability = exp(-energy_difference / (K_B * temperature));

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
        rc_radius * rc_radius * (scale_factor - 1) * (scale_factor - 1),
        rc_radius * rc_radius * scale_factor * scale_factor,
        atom.position_per_atom.data(),
        atom.position_per_atom.data() + atom.number_of_atoms,
        atom.position_per_atom.data() + atom.number_of_atoms * 2,
        outer_atoms_flags.data());
      gpuDeviceSynchronize();

      GPU_Vector<double> max_displacement(1);
      double average_displacement = get_outer_average_displacement(
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

      mc_output<< step << "\t" << max_value << "\t" << average_displacement 
      << "\t" << pe_before_total << "\t" << pe_after_total << "\t" << num_accepted / (double(step) + 1) << std::endl;

      //copy the relaxed local structure to the global structure
      build_all_atoms(
        atom,
        local_atoms,
        local_N_cpu,
        local_index.data());
    }
    //need modification
    force.potentials[0]->N2 = atom.number_of_atoms;
  }
}