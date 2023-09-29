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
The NVE ensemble integrator.
------------------------------------------------------------------------------*/

#include "ensemble_msst.cuh"
#include "utilities/common.cuh"
#define DIM 3

namespace
{
static __global__ void gpu_v2_sum(const int N, const double* g_vector, double* g_scalar)
{
  //<<<1, 1024>>>
  int tid = threadIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  double vector1, vector2, vector3;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N) {
      vector1 = g_vector[n];
      vector2 = g_vector[n + N];
      vector3 = g_vector[n + 2 * N];
      s_data[tid] += vector1 * vector1 + vector2 * vector2 + vector3 * vector3;
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
    g_scalar[0] = s_data[0];
  }
}

static __global__ void gpu_fun_1(
  const int N,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz,
  const double* g_mass,
  const double mu,
  const int shock_direction,
  const double* g_omega,
  const double* velocity_sum,
  const double volume,
  double* g_velocity,
  double dthalf)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    double mass = g_mass[n];
    double mass_inv = 1.0 / mass;
    double C[3] = {g_fx[n] * mass_inv, g_fy[n] * mass_inv, g_fz[n] * mass_inv};
    const double tmp =
      g_omega[shock_direction] * g_omega[shock_direction] * mu / (velocity_sum[0] * mass * volume);
    double D[3] = {tmp, tmp, tmp};
    D[shock_direction] -= 2.0 * g_omega[shock_direction] / volume;
    for (int i = 0; i < 3; i++) {
      if (fabs(dthalf * D[i]) > 1.0e-06) {
        const double expd = exp(D[i] * dthalf);
        g_velocity[n + i * N] = expd * (C[i] + D[i] * g_velocity[n + i * N] - C[i] / expd) / D[i];
      } else {
        g_velocity[n + i * N] =
          g_velocity[n + i * N] + (C[i] + D[i] * g_velocity[n + i * N]) * dthalf +
          0.5 * (D[i] * D[i] * g_velocity[n + i * N] + C[i] * D[i]) * dthalf * dthalf;
      }
    }
  }
}

static __global__ void
gpu_remap(const int N, const double dilation, double* g_position, double* g_velocity)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    g_position[n] = g_position[n] * dilation;
    g_velocity[n] = g_velocity[n] * dilation;
  }
}
} // namespace

Ensemble_MSST::Ensemble_MSST(const char** params, int num_params)
{
  // 0: ensemble
  // 1: msst
  // 2: x/y/z
  // 3: vs (km/h)
  // 4-5: q q_value
  // 6-7: mu mu_value
  // 8-9: tscale tscale_value (optional)

  if (strcmp(params[2], "x") == 0) {
    shock_direction = 0;
  } else if (strcmp(params[2], "y") == 0) {
    shock_direction = 1;
  } else if (strcmp(params[2], "z") == 0) {
    shock_direction = 2;
  } else {
    PRINT_INPUT_ERROR("Shock direction should be x or y or z.");
  }
  if (!is_valid_real(params[3], &shockvel))
    PRINT_INPUT_ERROR("Invalid shock velocity value.");
  if (!is_valid_real(params[5], &qmass))
    PRINT_INPUT_ERROR("Invalid qmass value.");
  if (!is_valid_real(params[7], &mu))
    PRINT_INPUT_ERROR("Invalid mu value.");
  if (num_params == 10) {
    if (!is_valid_real(params[9], &tscale)) {
      PRINT_INPUT_ERROR("Invalid tscale value.");
    }
  }
  // TODO: print summary

  dilation = 1.0;
}

Ensemble_MSST::~Ensemble_MSST(void)
{
  // nothing now
}

void Ensemble_MSST::find_thermo()
{
  Ensemble::find_thermo(
    false,
    box->get_volume(),
    *group,
    atom->mass,
    atom->potential_per_atom,
    atom->velocity_per_atom,
    atom->virial_per_atom,
    *thermo);
  thermo->copy_to_host(thermo_cpu.data());
  ke = 1.5 * kB * N * thermo_cpu[0];
  etotal = ke + thermo_cpu[1];
}

void Ensemble_MSST::init()
{
  N = atom->number_of_atoms;
  dthalf = time_step / 2;
  thermo_cpu.resize(thermo->size());
  find_thermo();
  v0 = box->get_volume();
  e0 = etotal;
  p0 = thermo_cpu[shock_direction + 4];
  printf("    MSST V0: %g A^3, E0: %g eV, P0: %g GPa\n", v0, e0, p0 * PRESSURE_UNIT_CONVERSION);

  lagrangian_position = 0.0;

  // compute total mass
  for (int i = 0; i < atom->cpu_mass.size(); i++)
    total_mass += atom->cpu_mass[i];

  double sqrt_initial_temperature_scaling = sqrt(1.0 - tscale);
  double fac1 = tscale * total_mass / qmass * thermo_cpu[0];
  omega = -1 * sqrt(fac1);
  double fac2 = omega / v0;

  scale_velocity_global(sqrt_initial_temperature_scaling, velocity_per_atom);
  /*gpu_scale_velocity<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, fac2, velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL*/
}

void Ensemble_MSST::get_vsum()
{
  GPU_Vector<double> v2sum;
  v2sum.resize(1);
  gpu_v2_sum<<<1, 1024>>>(N, atom->velocity_per_atom.data(), ret.data());
  v2sum.copy_to_host(&vsum);
}

void Ensemble_MSST::remap()
{
  box->cpu_h[shock_direction] *= dilation;
  box->cpu_h[shock_direction + 3] = 0.5 * box->cpu_h[shock_direction];
  gpu_remap<<<(N - 1) / 128 + 1, 128>>>(
    N,
    dilation,
    position_per_atom.data() + shock_direction * N,
    velocity_per_atom.data() + shock_direction * N);
  CUDA_CHECK_KERNEL
}

void Ensemble_MSST::get_e_scale()
{
  find_thermo();
  double volume = box->get_volume();

  // compute scalar
  double scalar = qmass * omega * omega / (2.0 * total_mass) -
                  0.5 * omega * shockvel * shockvel * (1.0 - volume / v0) * (1.0 - volume / v0) -
                  p0 * (v0 - volume);

  // conserved quantity
  double e_scale = etotal + scalar;
}

void Ensemble_MSST::get_omega()
{
  double volume = box->get_volume();
  // propagate the time derivative of the volume 1/2 step at fixed vol, r, rdot
  double p_msst = shockvel * shockvel * total_mass * (v0 - volume) / (v0 * v0);
  double A = total_mass * (thermo_cpu[shock_direction + 2] - p0 - p_msst) / qmass;
  double B = total_mass * mu / (qmass * volume);

  // prevent blow-up of the volume
  if (volume > v0 && A > 0.0)
    A = -A;

  // use Taylor expansion to avoid singularity at B = 0
  if (B * dthalf > 1.0e-06)
    omega = (omega + A * (exp(B * dthalf) - 1.0) / B) * exp(-B * dthalf);
  else
    omega = omega + (A - B * omega) * dthalf + 0.5 * (B * B * omega - A * B) * dthalf * dthalf;
}

void Ensemble_MSST::compute1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{

  get_e_scale();
  get_omega();

  // propagate velocity sum 1/2 step by temporarily propagating the velocities
  get_vsum();

  std::vector<double> cpu_old_velocity(velocity_per_atom.size());
  velocity_per_atom.copy_to_host(cpu_old_velocity.data());
  gpu_fun_1<<<(N - 1) / 128 + 1, 128>>>(
    N,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + 2 * N,
    mass.data(),
    mu,
    shock_direction,
    omega,
    velocity_sum,
    volume,
    velocity_per_atom.data(),
    dthalf);
  get_vsum();

  // reset the velocities
  velocity_per_atom.copy_from_host(cpu_old_velocity.data());

  // propagate velocities 1/2 step using the new velocity sum
  gpu_fun_1<<<(N - 1) / 128 + 1, 128>>>(
    N,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + 2 * N,
    mass.data(),
    mu,
    shock_direction,
    omega,
    velocity_sum,
    volume,
    velocity_per_atom.data(),
    dthalf);

  // propagate the volume 1/2 step
  double vol1 = volume + omega * dthalf;

  // rescale positions and change box size
  dilation = vol1 / volume;
  remap(N, box, position_per_atom, velocity_per_atom);

  velocity_verlet_x();

  // propagate the volume 1/2 step
  double vol2 = vol1 + omega * dthalf;

  // rescale positions and change box size
  dilation = vol2 / vol1;
  remap(N, box, position_per_atom, velocity_per_atom);
}

void Ensemble_MSST::compute2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{

  get_e_scale();

  // propagate particle velocities 1/2 step
  gpu_fun_1<<<(N - 1) / 128 + 1, 128>>>(
    N,
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + 2 * N,
    mass.data(),
    mu,
    shock_direction,
    omega,
    velocity_sum,
    volume,
    velocity_per_atom.data(),
    dthalf);

  // compute new pressure and volume
  find_thermo();
  // GPU_Vector<double> velocity_sum;
  // velocity_sum.resize(1);

  get_vsum();
  get_omega();

  // calculate Lagrangian position of computational cell
  lagrangian_position -= shockvel * volume / v0 * time_step;

  // velocity_verlet(
  //   false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  find_thermo();
}
