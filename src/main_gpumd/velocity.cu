/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Initialize the velocities of the system:
    total linear momentum is zero
    total angular momentum is zero
If DEBUG is on in the makefile, the velocities are the same from run to run.
If DEBUG is off, the velocities are different in different runs.
------------------------------------------------------------------------------*/

#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "velocity.cuh"
#include <algorithm>
#include <cstring>
#include <vector>

void Velocity::scale(
  const double initial_temperature,
  const std::vector<double>& cpu_mass,
  double* cpu_vx,
  double* cpu_vy,
  double* cpu_vz)
{
  const int N = cpu_mass.size();
  double temperature = 0.0;
  for (int n = 0; n < N; ++n) {
    double v2 = cpu_vx[n] * cpu_vx[n] + cpu_vy[n] * cpu_vy[n] + cpu_vz[n] * cpu_vz[n];
    temperature += cpu_mass[n] * v2;
  }
  temperature /= 3.0 * K_B * N;
  double factor = sqrt(initial_temperature / temperature);
  for (int n = 0; n < N; ++n) {
    cpu_vx[n] *= factor;
    cpu_vy[n] *= factor;
    cpu_vz[n] *= factor;
  }
}

static void get_random_velocities(const int N, double* vx, double* vy, double* vz)
{
  for (int n = 0; n < N; ++n) {
    vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
    vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
    vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
  }
}

static void get_random_velocities_by_seed(const int N, double* vx, double* vy, double* vz, int seed)
{
  unsigned int s = (unsigned int)seed;
  for (int n = 0; n < N; ++n) {
    srand(s + n * 3);
    vx[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
    srand(s + n * 3 + 1);
    vy[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
    srand(s + n * 3 + 2);
    vz[n] = -1.0 + (rand() * 2.0) / RAND_MAX;
  }
}

static void zero_linear_momentum(const int N, const double* m, double* vx, double* vy, double* vz)
{
  double center_of_mass_velocity[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < N; ++n) {
    total_mass += m[n];
    center_of_mass_velocity[0] += m[n] * vx[n];
    center_of_mass_velocity[1] += m[n] * vy[n];
    center_of_mass_velocity[2] += m[n] * vz[n];
  }
  center_of_mass_velocity[0] /= total_mass;
  center_of_mass_velocity[1] /= total_mass;
  center_of_mass_velocity[2] /= total_mass;
  for (int n = 0; n < N; ++n) {
    vx[n] -= center_of_mass_velocity[0];
    vy[n] -= center_of_mass_velocity[1];
    vz[n] -= center_of_mass_velocity[2];
  }
}

static void get_center(
  const int N, double r0[3], const double* m, const double* x, const double* y, const double* z)
{
  double mass_total = 0;
  for (int i = 0; i < N; i++) {
    double mass = m[i];
    mass_total += mass;
    r0[0] += x[i] * mass;
    r0[1] += y[i] * mass;
    r0[2] += z[i] * mass;
  }
  r0[0] /= mass_total;
  r0[1] /= mass_total;
  r0[2] /= mass_total;
}

static void get_angular_momentum(
  const int N,
  double L[3],
  const double r0[3],
  const double* m,
  const double* x,
  const double* y,
  const double* z,
  const double* vx,
  const double* vy,
  const double* vz)
{
  for (int i = 0; i < N; i++) {
    const double dx = x[i] - r0[0];
    const double dy = y[i] - r0[1];
    const double dz = z[i] - r0[2];
    L[0] += m[i] * (dy * vz[i] - dz * vy[i]);
    L[1] += m[i] * (dz * vx[i] - dx * vz[i]);
    L[2] += m[i] * (dx * vy[i] - dy * vx[i]);
  }
}

static void get_inertia(
  const int N,
  double I[3][3],
  const double r0[3],
  const double* m,
  const double* x,
  const double* y,
  const double* z)
{
  for (int i = 0; i < N; i++) {
    const double dx = x[i] - r0[0];
    const double dy = y[i] - r0[1];
    const double dz = z[i] - r0[2];
    I[0][0] += m[i] * (dy * dy + dz * dz);
    I[1][1] += m[i] * (dx * dx + dz * dz);
    I[2][2] += m[i] * (dx * dx + dy * dy);
    I[0][1] -= m[i] * dx * dy;
    I[1][2] -= m[i] * dy * dz;
    I[0][2] -= m[i] * dx * dz;
  }
  I[1][0] = I[0][1];
  I[2][1] = I[1][2];
  I[2][0] = I[0][2];
}

static __host__ __device__ double
get_angular_velocity(const double I[3][3], const double L[3], double w[3])
{
  double inverse[3][3]; // inverse of I
  inverse[0][0] = I[1][1] * I[2][2] - I[1][2] * I[2][1];
  inverse[0][1] = -(I[0][1] * I[2][2] - I[0][2] * I[2][1]);
  inverse[0][2] = I[0][1] * I[1][2] - I[0][2] * I[1][1];
  inverse[1][0] = -(I[1][0] * I[2][2] - I[1][2] * I[2][0]);
  inverse[1][1] = I[0][0] * I[2][2] - I[0][2] * I[2][0];
  inverse[1][2] = -(I[0][0] * I[1][2] - I[0][2] * I[1][0]);
  inverse[2][0] = I[1][0] * I[2][1] - I[1][1] * I[2][0];
  inverse[2][1] = -(I[0][0] * I[2][1] - I[0][1] * I[2][0]);
  inverse[2][2] = I[0][0] * I[1][1] - I[0][1] * I[1][0];
  double determinant = I[0][0] * I[1][1] * I[2][2] + I[0][1] * I[1][2] * I[2][0] +
                       I[0][2] * I[1][0] * I[2][1] - I[0][0] * I[1][2] * I[2][1] -
                       I[0][1] * I[1][0] * I[2][2] - I[2][0] * I[1][1] * I[0][2];
  if (determinant > -1.0e-10 && determinant < 1.0e-10) {
    w[0] = w[1] = w[2] = 0.0;
    return determinant;
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      inverse[i][j] /= determinant;
    }
  }
  // w = inv(I) * L, because L = I * w
  w[0] = inverse[0][0] * L[0] + inverse[0][1] * L[1] + inverse[0][2] * L[2];
  w[1] = inverse[1][0] * L[0] + inverse[1][1] * L[1] + inverse[1][2] * L[2];
  w[2] = inverse[2][0] * L[0] + inverse[2][1] * L[1] + inverse[2][2] * L[2];

  return determinant;
}

// v_i = v_i - w x dr_i
static void zero_angular_momentum(
  const int N,
  const double w[3],
  const double r0[3],
  const double* x,
  const double* y,
  const double* z,
  double* vx,
  double* vy,
  double* vz)
{
  for (int i = 0; i < N; i++) {
    const double dx = x[i] - r0[0];
    const double dy = y[i] - r0[1];
    const double dz = z[i] - r0[2];
    vx[i] -= w[1] * dz - w[2] * dy;
    vy[i] -= w[2] * dx - w[0] * dz;
    vz[i] -= w[0] * dy - w[1] * dx;
  }
}

void Velocity::correct_velocity(
  const int N,
  const std::vector<double>& cpu_mass,
  const std::vector<double>& cpu_position_per_atom,
  std::vector<double>& cpu_velocity_per_atom)
{
  zero_linear_momentum(
    N,
    cpu_mass.data(),
    cpu_velocity_per_atom.data(),
    cpu_velocity_per_atom.data() + N,
    cpu_velocity_per_atom.data() + N * 2);

  double r0[3] = {0, 0, 0}; // center of mass position
  get_center(
    N,
    r0,
    cpu_mass.data(),
    cpu_position_per_atom.data(),
    cpu_position_per_atom.data() + N,
    cpu_position_per_atom.data() + N * 2);

  double L[3] = {0, 0, 0}; // angular momentum
  get_angular_momentum(
    N,
    L,
    r0,
    cpu_mass.data(),
    cpu_position_per_atom.data(),
    cpu_position_per_atom.data() + N,
    cpu_position_per_atom.data() + N * 2,
    cpu_velocity_per_atom.data(),
    cpu_velocity_per_atom.data() + N,
    cpu_velocity_per_atom.data() + N * 2);

  double I[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}; // moment of inertia
  get_inertia(
    N,
    I,
    r0,
    cpu_mass.data(),
    cpu_position_per_atom.data(),
    cpu_position_per_atom.data() + N,
    cpu_position_per_atom.data() + N * 2);

  double w[3]; // angular velocity
  double determinant = get_angular_velocity(I, L, w);
  if (determinant > -1.0e-10 && determinant < 1.0e-10) {
    return; // do not correct the angular velocity to avoid NaN
  }

  zero_angular_momentum(
    N,
    w,
    r0,
    cpu_position_per_atom.data(),
    cpu_position_per_atom.data() + N,
    cpu_position_per_atom.data() + N * 2,
    cpu_velocity_per_atom.data(),
    cpu_velocity_per_atom.data() + N,
    cpu_velocity_per_atom.data() + N * 2);
}

static constexpr int CORRECTION_BLOCK_SIZE = 128;

template <int NUM_COMPONENTS>
static __device__ void sum_correction_components(double (&values)[NUM_COMPONENTS])
{
  __shared__ double s_values[NUM_COMPONENTS][CORRECTION_BLOCK_SIZE];
  const int tid = threadIdx.x;
  for (int d = 0; d < NUM_COMPONENTS; ++d) {
    s_values[d][tid] = values[d];
  }
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (tid < offset) {
      for (int d = 0; d < NUM_COMPONENTS; ++d) {
        s_values[d][tid] += s_values[d][tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    for (int d = 0; d < NUM_COMPONENTS; ++d) {
      values[d] = s_values[d][0];
    }
  }
}

static __global__ void find_center_sums(
  const int N,
  const int group_size,
  const int group_offset,
  const int* contents,
  const double* mass,
  const double* position,
  const double* velocity,
  double* sums)
{
  // Total mass, mass-weighted position, and linear momentum.
  double values[7] = {0.0};
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < group_size; k += blockDim.x * gridDim.x) {
    const int n = contents == nullptr ? k : contents[group_offset + k];
    const double m = mass[n];
    values[0] += m;
    for (int d = 0; d < 3; ++d) {
      values[1 + d] += m * position[n + d * N];
      values[4 + d] += m * velocity[n + d * N];
    }
  }
  sum_correction_components(values);
  if (threadIdx.x == 0) {
    for (int d = 0; d < 7; ++d) {
      sums[d * gridDim.x + blockIdx.x] = values[d];
    }
  }
}

static __global__ void
find_correction_center(const int num_blocks, const double* sums, double* parameters)
{
  double values[7] = {0.0};
  for (int k = threadIdx.x; k < num_blocks; k += blockDim.x) {
    for (int d = 0; d < 7; ++d) {
      values[d] += sums[d * num_blocks + k];
    }
  }
  sum_correction_components(values);
  if (threadIdx.x == 0) {
    for (int d = 0; d < 6; ++d) {
      parameters[d] = values[d + 1] / values[0];
    }
  }
}

static __global__ void find_rotation_sums(
  const int N,
  const int group_size,
  const int group_offset,
  const int* contents,
  const double* mass,
  const double* position,
  const double* velocity,
  const double* parameters,
  double* sums)
{
  // Six independent inertia components followed by angular momentum.
  double values[9] = {0.0};
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < group_size; k += blockDim.x * gridDim.x) {
    const int n = contents == nullptr ? k : contents[group_offset + k];
    const double m = mass[n];
    const double x = position[n] - parameters[0];
    const double y = position[n + N] - parameters[1];
    const double z = position[n + 2 * N] - parameters[2];
    const double vx = velocity[n] - parameters[3];
    const double vy = velocity[n + N] - parameters[4];
    const double vz = velocity[n + 2 * N] - parameters[5];
    values[0] += m * (y * y + z * z);
    values[1] += m * (x * x + z * z);
    values[2] += m * (x * x + y * y);
    values[3] -= m * x * y;
    values[4] -= m * y * z;
    values[5] -= m * x * z;
    values[6] += m * (y * vz - z * vy);
    values[7] += m * (z * vx - x * vz);
    values[8] += m * (x * vy - y * vx);
  }
  sum_correction_components(values);
  if (threadIdx.x == 0) {
    for (int d = 0; d < 9; ++d) {
      sums[d * gridDim.x + blockIdx.x] = values[d];
    }
  }
}

static __global__ void
find_correction_rotation(const int num_blocks, const double* sums, double* parameters)
{
  double values[9] = {0.0};
  for (int k = threadIdx.x; k < num_blocks; k += blockDim.x) {
    for (int d = 0; d < 9; ++d) {
      values[d] += sums[d * num_blocks + k];
    }
  }
  sum_correction_components(values);
  if (threadIdx.x == 0) {
    const double I[3][3] = {
      {values[0], values[3], values[5]},
      {values[3], values[1], values[4]},
      {values[5], values[4], values[2]}};
    get_angular_velocity(I, values + 6, parameters + 6);
  }
}

static __global__ void apply_velocity_correction(
  const int N,
  const int group_size,
  const int group_offset,
  const int* contents,
  const double* position,
  const double* parameters,
  double* velocity)
{
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < group_size; k += blockDim.x * gridDim.x) {
    const int n = contents == nullptr ? k : contents[group_offset + k];
    const double x = position[n] - parameters[0];
    const double y = position[n + N] - parameters[1];
    const double z = position[n + 2 * N] - parameters[2];
    velocity[n] -= parameters[3];
    velocity[n + N] -= parameters[4];
    velocity[n + 2 * N] -= parameters[5];
    velocity[n] -= parameters[7] * z - parameters[8] * y;
    velocity[n + N] -= parameters[8] * x - parameters[6] * z;
    velocity[n + 2 * N] -= parameters[6] * y - parameters[7] * x;
  }
}

void Velocity::correct_velocity(const int step, const std::vector<Group>& group, Atom& atom)
{
  const int N = atom.number_of_atoms;
  if (!do_velocity_correction || step % velocity_correction_interval != 0 || N == 0) {
    return;
  }

  const Group* selected_group =
    velocity_correction_group_method < 0 ? nullptr : &group[velocity_correction_group_method];
  const int num_groups = selected_group == nullptr ? 1 : selected_group->number;
  const int* contents = selected_group == nullptr ? nullptr : selected_group->contents.data();
  const int max_blocks = std::min((N - 1) / CORRECTION_BLOCK_SIZE + 1, 1024);
  if (correction_sums.size() < 9 * max_blocks) {
    correction_sums.resize(9 * max_blocks);
  }
  if (correction_parameters.size() == 0) {
    correction_parameters.resize(9);
  }

  for (int g = 0; g < num_groups; ++g) {
    const int group_size = selected_group == nullptr ? N : selected_group->cpu_size[g];
    const int group_offset = selected_group == nullptr ? 0 : selected_group->cpu_size_sum[g];
    if (group_size == 0) {
      continue;
    }
    const int num_blocks = std::min((group_size - 1) / CORRECTION_BLOCK_SIZE + 1, max_blocks);
    find_center_sums<<<num_blocks, CORRECTION_BLOCK_SIZE>>>(
      N,
      group_size,
      group_offset,
      contents,
      atom.mass.data(),
      atom.position_per_atom.data(),
      atom.velocity_per_atom.data(),
      correction_sums.data());
    GPU_CHECK_KERNEL
    find_correction_center<<<1, CORRECTION_BLOCK_SIZE>>>(
      num_blocks, correction_sums.data(), correction_parameters.data());
    GPU_CHECK_KERNEL
    find_rotation_sums<<<num_blocks, CORRECTION_BLOCK_SIZE>>>(
      N,
      group_size,
      group_offset,
      contents,
      atom.mass.data(),
      atom.position_per_atom.data(),
      atom.velocity_per_atom.data(),
      correction_parameters.data(),
      correction_sums.data());
    GPU_CHECK_KERNEL
    find_correction_rotation<<<1, CORRECTION_BLOCK_SIZE>>>(
      num_blocks, correction_sums.data(), correction_parameters.data());
    GPU_CHECK_KERNEL
    apply_velocity_correction<<<num_blocks, CORRECTION_BLOCK_SIZE>>>(
      N,
      group_size,
      group_offset,
      contents,
      atom.position_per_atom.data(),
      correction_parameters.data(),
      atom.velocity_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

void Velocity::initialize_cpu(
  const bool has_velocity_in_xyz,
  const double initial_temperature,
  Atom& atom,
  bool use_seed,
  int seed)
{
  const int N = atom.number_of_atoms;

  do_velocity_correction = false;
  if (!has_velocity_in_xyz) {
    if (use_seed) {
      get_random_velocities_by_seed(
        N,
        atom.cpu_velocity_per_atom.data(),
        atom.cpu_velocity_per_atom.data() + N,
        atom.cpu_velocity_per_atom.data() + N * 2,
        seed);
    } else {
      get_random_velocities(
        N,
        atom.cpu_velocity_per_atom.data(),
        atom.cpu_velocity_per_atom.data() + N,
        atom.cpu_velocity_per_atom.data() + N * 2);
    }
    correct_velocity(N, atom.cpu_mass, atom.cpu_position_per_atom, atom.cpu_velocity_per_atom);
    scale(
      initial_temperature,
      atom.cpu_mass,
      atom.cpu_velocity_per_atom.data(),
      atom.cpu_velocity_per_atom.data() + N,
      atom.cpu_velocity_per_atom.data() + N * 2);
  }
}

void Velocity::finalize()
{
  do_velocity_correction = false;
  velocity_correction_group_method = -1;
}
