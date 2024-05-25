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

#include "ensemble_mirror.cuh"

namespace
{
static __global__ void
gpu_find_wall(int number_of_atoms, double right, bool* right_wall_list, double* position)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms)
    right_wall_list[i] = (position[i] > right);
}

static __global__ void gpu_velocity_verlet(
  const bool is_step1,
  const int number_of_particles,
  const double mirror_vel,
  const double mirror_pos,
  const double g_time_step,
  bool* right_wall_list,
  const double* g_mass,
  double* g_x,
  double* g_y,
  double* g_z,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double time_step = g_time_step;
    const double time_step_half = time_step * 0.5;
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    const double mass_inv = 1.0 / g_mass[i];
    const double ax = g_fx[i] * mass_inv;
    const double ay = g_fy[i] * mass_inv;
    const double az = g_fz[i] * mass_inv;
    if (right_wall_list[i]) {
      vx = 0;
      vy = 0;
      vz = 0;
      g_vx[i] = 0.0;
      g_vy[i] = 0.0;
      g_vz[i] = 0.0;
    } else {
      vx += ax * time_step_half;
      vy += ay * time_step_half;
      vz += az * time_step_half;
      g_vx[i] = vx;
      g_vy[i] = vy;
      g_vz[i] = vz;
    }

    if (is_step1) {
      g_x[i] += vx * time_step;
      g_y[i] += vy * time_step;
      g_z[i] += vz * time_step;
    } else if (g_x[i] < mirror_pos) {
      g_x[i] = 2 * mirror_pos - g_x[i];
      g_vx[i] = 2 * mirror_vel - g_vx[i];
    }
  }
}
} // namespace

Ensemble_mirror::Ensemble_mirror(const char** params, int num_params)
{
  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "thickness") == 0) {
      if (!is_valid_real(params[i + 1], &thickness))
        PRINT_INPUT_ERROR("Wrong inputs for thickness keyword.");
      i += 2;
    } else if (strcmp(params[i], "vp") == 0) {
      if (!is_valid_real(params[i + 1], &vp))
        PRINT_INPUT_ERROR("Wrong inputs for vp keyword.");
      i += 2;
    } else if (strcmp(params[i], "accerlate") == 0) {
      if (!is_valid_int(params[i + 1], &accerlate_steps))
        PRINT_INPUT_ERROR("Wrong inputs for vp keyword.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }
  printf("Piston velocity: %f km/s.\n", vp);
  printf("The thickness of fixed wall on the right side: %f Ang.\n", thickness);
  vp = vp / 100 * TIME_UNIT_CONVERSION;
}

void Ensemble_mirror::init()
{
  int N = atom->number_of_atoms;
  gpu_right_wall_list.resize(N, false);
  gpu_find_wall<<<(N - 1) / 128 + 1, 128>>>(
    N, box->cpu_h[0] - thickness, gpu_right_wall_list.data(), atom->position_per_atom.data());
}

Ensemble_mirror::~Ensemble_mirror(void) {}

void Ensemble_mirror::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0)
    init();
  find_thermo(
    false,
    box.get_volume(),
    group,
    atoms.mass,
    atoms.potential_per_atom,
    atoms.velocity_per_atom,
    atoms.virial_per_atom,
    thermo);
  int n = atoms.number_of_atoms;
  gpu_velocity_verlet<<<(n - 1) / 128 + 1, 128>>>(
    true,
    n,
    vp,
    mirror_pos,
    time_step,
    gpu_right_wall_list.data(),
    atoms.mass.data(),
    atoms.position_per_atom.data(),
    atoms.position_per_atom.data() + n,
    atoms.position_per_atom.data() + 2 * n,
    atoms.velocity_per_atom.data(),
    atoms.velocity_per_atom.data() + n,
    atoms.velocity_per_atom.data() + 2 * n,
    atoms.force_per_atom.data(),
    atoms.force_per_atom.data() + n,
    atoms.force_per_atom.data() + 2 * n);
}

void Ensemble_mirror::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  int n = atoms.number_of_atoms;
  if (*current_step < accerlate_steps)
    vp_current = vp * ((double)*current_step / (double)accerlate_steps);
  else
    vp_current = vp;

  mirror_pos += time_step * vp_current;
  gpu_velocity_verlet<<<(n - 1) / 128 + 1, 128>>>(
    false,
    n,
    vp,
    mirror_pos,
    time_step,
    gpu_right_wall_list.data(),
    atoms.mass.data(),
    atoms.position_per_atom.data(),
    atoms.position_per_atom.data() + n,
    atoms.position_per_atom.data() + 2 * n,
    atoms.velocity_per_atom.data(),
    atoms.velocity_per_atom.data() + n,
    atoms.velocity_per_atom.data() + 2 * n,
    atoms.force_per_atom.data(),
    atoms.force_per_atom.data() + n,
    atoms.force_per_atom.data() + 2 * n);
}
