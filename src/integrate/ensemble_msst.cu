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
The NVE ensemble integrator.
------------------------------------------------------------------------------*/

#include "ensemble_msst.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

namespace
{
static __global__ void gpu_get_vsum(const int N, const double* g_vector, double* g_scalar)
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

static __global__ void gpu_msst_v(
  const int N,
  const double* g_f,
  double* g_velocity,
  const double* g_mass,
  const double mu,
  const int shock_direction,
  const double omega,
  const double vsum,
  const double volume,
  double dthalf)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    double mass = g_mass[n];
    double mass_inv = 1.0 / mass;
    double C[3] = {g_f[n] * mass_inv, g_f[n + N] * mass_inv, g_f[n + 2 * N] * mass_inv};
    const double tmp = omega * omega * mu / (vsum * mass * volume);
    double D[3] = {tmp, tmp, tmp};
    D[shock_direction] -= 2.0 * omega / volume;
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
  if (!is_valid_real(params[3], &vs))
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
  printf(
    "Performing MSST simulation in direction %d with shock velocity = %f, qmass = %f, mu = %f\n",
    shock_direction,
    vs,
    qmass,
    mu);
  vs *= 0.01;
  vs *= TIME_UNIT_CONVERSION;
  gpu_vsum.resize(1);
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
  vol = box->get_volume();
  p_current = thermo_cpu[shock_direction + 2];
}

void Ensemble_MSST::init()
{
  N = atom->number_of_atoms;
  dthalf = time_step / 2;
  thermo_cpu.resize(thermo->size());
  gpu_v_backup.resize(atom->cpu_velocity_per_atom.size());
  find_thermo();
  v0 = vol;
  e0 = etotal;
  p0 = p_current;
  printf("    MSST V0: %g A^3, E0: %g eV, P0: %g GPa\n", v0, e0, p0 * PRESSURE_UNIT_CONVERSION);

  // compute total mass
  for (int i = 0; i < atom->cpu_mass.size(); i++)
    total_mass += atom->cpu_mass[i];

  omega = -sqrt(tscale * total_mass / qmass * ke);

  scale_velocity_global(sqrt(1.0 - tscale), atom->velocity_per_atom);

  printf("    Initial strain rate %f, reduce temperature by %f\n", omega / v0, tscale);
}

void Ensemble_MSST::get_vsum()
{
  gpu_get_vsum<<<1, 1024>>>(N, atom->velocity_per_atom.data(), gpu_vsum.data());
  gpu_vsum.copy_to_host(&vsum);
}

void Ensemble_MSST::remap(double dilation)
{
  box->cpu_h[shock_direction] *= dilation;
  box->cpu_h[shock_direction + 3] = 0.5 * box->cpu_h[shock_direction];
  gpu_remap<<<(N - 1) / 128 + 1, 128>>>(
    N,
    dilation,
    atom->position_per_atom.data() + shock_direction * N,
    atom->velocity_per_atom.data() + shock_direction * N);
  GPU_CHECK_KERNEL
}

void Ensemble_MSST::get_conserved()
{
  find_thermo();

  // compute msst energy
  e_msst = 0.5 * qmass * omega * omega / total_mass;
  e_msst -= 0.5 * total_mass * vs * vs * (1.0 - vol / v0) * (1.0 - vol / v0);
  e_msst -= p0 * (v0 - vol);

  // conserved quantity
  e_conserved = etotal + e_msst;

  dhugo = (0.5 * (p_current + p0) * (v0 - vol)) + e0 - etotal;
  dray = p_current - p0 - total_mass * vs * vs * (1.0 - vol / v0) / v0;

  e_conserved /= N;
  dhugo /= 3 * N * kB;
  dray *= PRESSURE_UNIT_CONVERSION;
}

void Ensemble_MSST::get_omega()
{
  // propagate the time derivative of the volume 1/2 step at fixed vol, r, rdot
  double p_msst = vs * vs * total_mass * (v0 - vol) / (v0 * v0);
  double A = total_mass * (p_current - p0 - p_msst) / qmass;
  double B = total_mass * mu / (qmass * vol);

  // prevent blow-up of the volume
  if (vol > v0 && A > 0.0)
    A = -A;

  // use Taylor expansion to avoid singularity at B = 0
  if (B * dthalf > 1.0e-06)
    omega = (omega + A * (exp(B * dthalf) - 1.0) / B) * exp(-B * dthalf);
  else
    omega = omega + (A - B * omega) * dthalf + 0.5 * (B * B * omega - A * B) * dthalf * dthalf;
}

void Ensemble_MSST::msst_v()
{
  // propagate particle velocities 1/2 step
  gpu_msst_v<<<(N - 1) / 128 + 1, 128>>>(
    N,
    atom->force_per_atom.data(),
    atom->velocity_per_atom.data(),
    atom->mass.data(),
    mu,
    shock_direction,
    omega,
    vsum,
    vol,
    dthalf);
}

void Ensemble_MSST::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0)
    init();

  get_conserved();
  get_omega();

  get_vsum();
  CHECK(gpuMemcpy(
    gpu_v_backup.data(),
    atom.velocity_per_atom.data(),
    sizeof(double) * gpu_v_backup.size(),
    gpuMemcpyDeviceToDevice));

  // propagate velocity sum 1/2 step by temporarily propagating the velocities
  msst_v();
  get_vsum();

  // reset the velocities
  CHECK(gpuMemcpy(
    atom.velocity_per_atom.data(),
    gpu_v_backup.data(),
    sizeof(double) * gpu_v_backup.size(),
    gpuMemcpyDeviceToDevice));

  // propagate velocities 1/2 step using the new velocity sum
  msst_v();

  // propagate the volume 1/2 step
  double vol1 = vol + omega * dthalf;

  // rescale positions and change box size
  remap(vol1 / vol);

  velocity_verlet_x();

  // propagate the volume 1/2 step
  double vol2 = vol1 + omega * dthalf;

  // rescale positions and change box size
  remap(vol2 / vol1);

  if (*current_step == 0 || *current_step % (*total_steps / 10) == 0) {
    printf(
      "    MSST conserved energy: %f eV/atom, dHugoniot: %f K, dRayleigh: %f GPa\n",
      e_conserved,
      dhugo,
      dray);
  }
}

void Ensemble_MSST::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  get_conserved();
  msst_v();
  find_thermo();
  get_vsum();
  get_omega();

  // calculate Lagrangian position of computational cell
  lagrangian_position -= vs * vol / v0 * time_step;
}