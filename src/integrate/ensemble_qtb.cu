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
The QTB thermostat based on a colored noise filter:
[1] Dammak, T., et al. Phys. Rev. Lett. 103, 190601 (2009).
------------------------------------------------------------------------------*/

#include "ensemble_qtb.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdlib>

namespace
{
static __global__ void gpu_initialize_qtb_history(
  gpurandState* states,
  const int N,
  const int nfreq2,
  double* random_array_0,
  double* random_array_1,
  double* random_array_2)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = states[n];
    const int offset = n * nfreq2;
    for (int m = 0; m < nfreq2; ++m) {
      random_array_0[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
      random_array_1[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
      random_array_2[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
    }
    states[n] = state;
  }
}

static __global__ void gpu_refresh_qtb_random_force(
  gpurandState* states,
  const int N,
  const int nfreq2,
  const double* time_H,
  const double gamma3_prefactor,
  const double* mass,
  double* random_array_0,
  double* random_array_1,
  double* random_array_2,
  double* fran_x,
  double* fran_y,
  double* fran_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = states[n];
    const int offset = n * nfreq2;

    for (int m = 0; m < nfreq2 - 1; ++m) {
      random_array_0[offset + m] = random_array_0[offset + m + 1];
      random_array_1[offset + m] = random_array_1[offset + m + 1];
      random_array_2[offset + m] = random_array_2[offset + m + 1];
    }
    random_array_0[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);
    random_array_1[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);
    random_array_2[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);

    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_z = 0.0;
    for (int m = 0; m < nfreq2; ++m) {
      const int reverse_index = offset + nfreq2 - m - 1;
      const double h = time_H[m];
      sum_x += h * random_array_0[reverse_index];
      sum_y += h * random_array_1[reverse_index];
      sum_z += h * random_array_2[reverse_index];
    }

    const double gamma3 = gamma3_prefactor * sqrt(mass[n]);
    fran_x[n] = sum_x * gamma3;
    fran_y[n] = sum_y * gamma3;
    fran_z[n] = sum_z * gamma3;

    states[n] = state;
  }
}

static __global__ void gpu_apply_qtb_half_step(
  const int N,
  const double dt_half,
  const double fric_coef,
  const double* mass,
  const double* fran_x,
  const double* fran_y,
  const double* fran_z,
  double* vx,
  double* vy,
  double* vz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double mass_inv = 1.0 / mass[n];
    vx[n] += dt_half * (fran_x[n] * mass_inv - fric_coef * vx[n]);
    vy[n] += dt_half * (fran_y[n] * mass_inv - fric_coef * vy[n]);
    vz[n] += dt_half * (fran_z[n] * mass_inv - fric_coef * vz[n]);
  }
}
} // namespace

Ensemble_QTB::Ensemble_QTB(
  int t,
  int N,
  double T,
  double Tc,
  double dt_input,
  double f_max_input,
  int N_f_input,
  int seed_input)
{
  type = t;
  number_of_atoms = N;
  temperature = T;
  temperature_coupling = Tc;
  dt = dt_input;
  seed = seed_input;
  N_f = N_f_input;
  nfreq2 = 2 * N_f;

  // Input f_max uses ps^-1. Convert to internal time unit.
  f_max_natural = f_max_input * TIME_UNIT_CONVERSION / 1000.0;

  int alpha_estimate = int(1.0 / (2.0 * f_max_natural * dt));
  if (alpha_estimate < 1) {
    alpha = 1;
  } else {
    alpha = alpha_estimate;
  }

  h_timestep = alpha * dt;
  fric_coef = 1.0 / (temperature_coupling * dt);
  counter_mu = 0;
  last_filter_temperature = -1.0;

  time_H_host.resize(nfreq2, 0.0);
  time_H_device.resize(nfreq2);

  random_array_0.resize(size_t(number_of_atoms) * size_t(nfreq2));
  random_array_1.resize(size_t(number_of_atoms) * size_t(nfreq2));
  random_array_2.resize(size_t(number_of_atoms) * size_t(nfreq2));
  fran.resize(size_t(number_of_atoms) * 3);

  curand_states.resize(number_of_atoms);
  initialize_curand_states<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(), number_of_atoms, seed);
  GPU_CHECK_KERNEL

  gpu_initialize_qtb_history<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(),
    number_of_atoms,
    nfreq2,
    random_array_0.data(),
    random_array_1.data(),
    random_array_2.data());
  GPU_CHECK_KERNEL
}

Ensemble_QTB::~Ensemble_QTB(void)
{
  // nothing
}

void Ensemble_QTB::update_time_filter(const double target_temperature)
{
  if (fabs(target_temperature - last_filter_temperature) < 1.0e-12) {
    return;
  }

  std::vector<double> omega_H(nfreq2, 0.0);

  for (int k = 0; k < nfreq2; ++k) {
    const double k_shift = k - N_f;
    const double f_k = k_shift / (nfreq2 * h_timestep);

    if (k == N_f) {
      omega_H[k] = sqrt(K_B * target_temperature);
      continue;
    }

    const double energy_k = 2.0 * PI * HBAR * fabs(f_k);
    const double x = energy_k / (K_B * target_temperature);
    double qfactor = 0.5;
    if (x < 200.0) {
      qfactor += 1.0 / (exp(x) - 1.0);
    }

    omega_H[k] = sqrt(energy_k * qfactor);
    const double numerator = sin(k_shift * PI / (2.0 * alpha * N_f));
    const double denominator = sin(k_shift * PI / (2.0 * N_f));
    omega_H[k] *= alpha * numerator / denominator;
  }

  for (int n = 0; n < nfreq2; ++n) {
    double value = 0.0;
    const double t_n = n - N_f;
    for (int k = 0; k < nfreq2; ++k) {
      const double omega_k = (k - N_f) * PI / N_f;
      value += omega_H[k] * cos(omega_k * t_n);
    }
    time_H_host[n] = value / nfreq2;
  }

  time_H_device.copy_from_host(time_H_host.data());
  last_filter_temperature = target_temperature;
}

void Ensemble_QTB::refresh_colored_random_force()
{
  const double gamma3_prefactor = sqrt(2.0 * fric_coef * 12.0 / h_timestep);
  gpu_refresh_qtb_random_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(),
    number_of_atoms,
    nfreq2,
    time_H_device.data(),
    gamma3_prefactor,
    atom->mass.data(),
    random_array_0.data(),
    random_array_1.data(),
    random_array_2.data(),
    fran.data(),
    fran.data() + number_of_atoms,
    fran.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL
}

void Ensemble_QTB::apply_qtb_half_step()
{
  const double dt_half = 0.5 * dt;

  gpu_apply_qtb_half_step<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    dt_half,
    fric_coef,
    atom->mass.data(),
    fran.data(),
    fran.data() + number_of_atoms,
    fran.data() + 2 * number_of_atoms,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL

  gpu_find_momentum<<<4, 1024>>>(
    number_of_atoms,
    atom->mass.data(),
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL

  gpu_correct_momentum<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + number_of_atoms,
    atom->velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

void Ensemble_QTB::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (counter_mu == 0) {
    update_time_filter(temperature);
    refresh_colored_random_force();
  }

  apply_qtb_half_step();

#ifdef USE_NEPCG
  velocity_verlet_cg(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#else
  velocity_verlet(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#endif
}

void Ensemble_QTB::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
#ifdef USE_NEPCG
  velocity_verlet_cg(
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#else
  velocity_verlet(
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
#endif

  apply_qtb_half_step();

  find_thermo(
    true,
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);

  counter_mu = (counter_mu + 1) % alpha;
}
