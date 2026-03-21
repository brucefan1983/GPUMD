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
NPH + QTB: Parrinello-Rahman barostat (MTTK) with QTB colored noise thermostat.
Equivalent to LAMMPS fix nph + fix qtb.
[1] Dammak, T., et al. Phys. Rev. Lett. 103, 190601 (2009).
[2] Martyna, G. J., et al. J. Chem. Phys. 101, 4177 (1994).
------------------------------------------------------------------------------*/

#include "ensemble_nph_qtb.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstring>

/* PLACEHOLDER_KERNELS */

namespace
{
static __global__ void gpu_initialize_qtb_history(
  gpurandState* states, const int N, const int nfreq2,
  double* r0, double* r1, double* r2)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = states[n];
    const int offset = n * nfreq2;
    for (int m = 0; m < nfreq2; ++m) {
      r0[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
      r1[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
      r2[offset + m] = gpurand_normal_double(&state) / sqrt(12.0);
    }
    states[n] = state;
  }
}

static __global__ void gpu_refresh_qtb_random_force(
  gpurandState* states, const int N, const int nfreq2,
  const double* time_H, const double gamma3_prefactor, const double* mass,
  double* r0, double* r1, double* r2,
  double* fran_x, double* fran_y, double* fran_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = states[n];
    const int offset = n * nfreq2;
    for (int m = 0; m < nfreq2 - 1; ++m) {
      r0[offset + m] = r0[offset + m + 1];
      r1[offset + m] = r1[offset + m + 1];
      r2[offset + m] = r2[offset + m + 1];
    }
    r0[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);
    r1[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);
    r2[offset + nfreq2 - 1] = gpurand_normal_double(&state) / sqrt(12.0);

    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int m = 0; m < nfreq2; ++m) {
      const int ri = offset + nfreq2 - m - 1;
      const double h = time_H[m];
      sx += h * r0[ri]; sy += h * r1[ri]; sz += h * r2[ri];
    }
    const double g3 = gamma3_prefactor * sqrt(mass[n]);
    fran_x[n] = sx * g3; fran_y[n] = sy * g3; fran_z[n] = sz * g3;
    states[n] = state;
  }
}

static __global__ void gpu_apply_qtb_half_step(
  const int N, const double dt_half, const double fric_coef,
  const double* mass, const double* fx, const double* fy, const double* fz,
  double* vx, double* vy, double* vz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double mi = 1.0 / mass[n];
    vx[n] += dt_half * (fx[n] * mi - fric_coef * vx[n]);
    vy[n] += dt_half * (fy[n] * mi - fric_coef * vy[n]);
    vz[n] += dt_half * (fz[n] * mi - fric_coef * vz[n]);
  }
}
} // namespace

/* PLACEHOLDER_CONSTRUCTOR */

Ensemble_NPH_QTB::Ensemble_NPH_QTB(const char** params, int num_params)
{
  // Initialize MTTK matrices to zero (same as parent constructor)
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      h[i][j] = h_inv[i][j] = h_old[i][j] = h_old_inv[i][j] = tmp1[i][j] = tmp2[i][j] =
        sigma[i][j] = f_deviatoric[i][j] = p_start[i][j] = p_stop[i][j] = p_current[i][j] =
          p_target[i][j] = p_hydro[i][j] = p_freq[i][j] = omega_dot[i][j] = omega_mass[i][j] =
            p_flag[i][j] = h_ref_inv[i][j] = 0;
      p_period[i][j] = 1000;
      need_scale[i][j] = true;
    }
  }

  // NPH + QTB: barostat on, NHC thermostat off (QTB replaces it)
  ensemble_type = NPH;
  use_barostat = true;
  use_thermostat = false;

  // QTB defaults
  qtb_f_max = 200.0;
  int qtb_n_f_input = 100;
  qtb_seed = 880302;

  // Parse parameters: nph_qtb <pressure_args> temp <T1> <T2> tperiod <tp> [f_max ...] [N_f ...] [seed ...]
  int i = 2; // skip "ensemble" and "nph_qtb"
  while (i < num_params) {
    if (strcmp(params[i], "iso") == 0 || strcmp(params[i], "aniso") == 0 ||
        strcmp(params[i], "tri") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start.");
      p_start[1][1] = p_start[2][2] = p_start[0][0];
      if (!is_valid_real(params[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop.");
      p_stop[1][1] = p_stop[2][2] = p_stop[0][0];
      p_flag[0][0] = p_flag[1][1] = p_flag[2][2] = true;
      if (strcmp(params[i], "iso") == 0)
        couple_type = XYZ;
      if (strcmp(params[i], "tri") == 0) {
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++)
            if (a != b) { p_start[a][b] = 0; p_stop[a][b] = 0; p_flag[a][b] = true; need_scale[a][b] = false; }
      }
      i += 3;
    } else if (strcmp(params[i], "x") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[0][0])) PRINT_INPUT_ERROR("Wrong p_start for x.");
      if (!is_valid_real(params[i + 2], &p_stop[0][0])) PRINT_INPUT_ERROR("Wrong p_stop for x.");
      p_flag[0][0] = 1; non_hydrostatic = 1; i += 3;
    } else if (strcmp(params[i], "y") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[1][1])) PRINT_INPUT_ERROR("Wrong p_start for y.");
      if (!is_valid_real(params[i + 2], &p_stop[1][1])) PRINT_INPUT_ERROR("Wrong p_stop for y.");
      p_flag[1][1] = 1; non_hydrostatic = 1; i += 3;
    } else if (strcmp(params[i], "z") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[2][2])) PRINT_INPUT_ERROR("Wrong p_start for z.");
      if (!is_valid_real(params[i + 2], &p_stop[2][2])) PRINT_INPUT_ERROR("Wrong p_stop for z.");
      p_flag[2][2] = 1; non_hydrostatic = 1; i += 3;
    } else if (strcmp(params[i], "pperiod") == 0) {
      if (!is_valid_real(params[i + 1], &p_period[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for pperiod.");
      if (p_period[0][0] < 200)
        PRINT_INPUT_ERROR("pperiod should >= 200 timestep.");
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          p_period[a][b] = p_period[0][0];
      i += 2;
    } else if (strcmp(params[i], "temp") == 0) {
      if (!is_valid_real(params[i + 1], &t_start)) PRINT_INPUT_ERROR("Wrong t_start.");
      if (!is_valid_real(params[i + 2], &t_stop)) PRINT_INPUT_ERROR("Wrong t_stop.");
      t_target = t_start;
      i += 3;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (!is_valid_real(params[i + 1], &t_period)) PRINT_INPUT_ERROR("Wrong tperiod.");
      i += 2;
    } else if (strcmp(params[i], "f_max") == 0) {
      if (!is_valid_real(params[i + 1], &qtb_f_max)) PRINT_INPUT_ERROR("f_max should be a number.");
      i += 2;
    } else if (strcmp(params[i], "N_f") == 0) {
      if (!is_valid_int(params[i + 1], &qtb_n_f_input)) PRINT_INPUT_ERROR("N_f should be an integer.");
      i += 2;
    } else if (strcmp(params[i], "seed") == 0) {
      if (!is_valid_int(params[i + 1], &qtb_seed)) PRINT_INPUT_ERROR("seed should be an integer.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown nph_qtb keyword.");
    }
  }

  if (t_start <= 0 || t_stop <= 0)
    PRINT_INPUT_ERROR("nph_qtb requires temp <T_start> <T_stop>.");
  if (!use_barostat)
    PRINT_INPUT_ERROR("nph_qtb requires pressure specification (iso/aniso/tri/x/y/z).");

  qtb_N_f = qtb_n_f_input;

  // Print summary
  printf("Use NPH + QTB ensemble for this run.\n");
  printf("    Parrinello-Rahman barostat + quantum thermal bath thermostat.\n");
  printf("    QTB temperature: t_start=%g K, t_stop=%g K\n", t_start, t_stop);
  printf("    QTB tperiod=%g timesteps\n", t_period);
  printf("    QTB f_max=%g ps^-1, N_f=%d, seed=%d\n", qtb_f_max, qtb_N_f, qtb_seed);

  const char* sc[3][3] = {{"xx","xy","xz"},{"yx","yy","yz"},{"zx","zy","zz"}};
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++)
      if (p_flag[a][b])
        printf("    %s: p_start=%g, p_stop=%g, pperiod=%g\n", sc[a][b], p_start[a][b], p_stop[a][b], p_period[a][b]);
}

Ensemble_NPH_QTB::~Ensemble_NPH_QTB(void) {}

/* PLACEHOLDER_INIT */

void Ensemble_NPH_QTB::init_mttk()
{
  // Call parent init for barostat setup
  Ensemble_MTTK::init_mttk();
  // Then initialize QTB
  init_qtb();
}

void Ensemble_NPH_QTB::init_qtb()
{
  qtb_number_of_atoms = atom->number_of_atoms;
  qtb_dt = time_step;
  qtb_nfreq2 = 2 * qtb_N_f;

  qtb_f_max_natural = qtb_f_max * TIME_UNIT_CONVERSION / 1000.0;
  int alpha_est = int(1.0 / (2.0 * qtb_f_max_natural * qtb_dt));
  qtb_alpha = (alpha_est < 1) ? 1 : alpha_est;

  qtb_h_timestep = qtb_alpha * qtb_dt;
  qtb_fric_coef = 1.0 / (t_period * qtb_dt);
  qtb_counter_mu = 0;
  qtb_last_filter_temperature = -1.0;

  qtb_time_H_host.resize(qtb_nfreq2, 0.0);
  qtb_time_H_device.resize(qtb_nfreq2);

  size_t sz = size_t(qtb_number_of_atoms) * size_t(qtb_nfreq2);
  qtb_random_array_0.resize(sz);
  qtb_random_array_1.resize(sz);
  qtb_random_array_2.resize(sz);
  qtb_fran.resize(size_t(qtb_number_of_atoms) * 3);

  qtb_curand_states.resize(qtb_number_of_atoms);
  initialize_curand_states<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_curand_states.data(), qtb_number_of_atoms, qtb_seed);
  GPU_CHECK_KERNEL

  gpu_initialize_qtb_history<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_curand_states.data(), qtb_number_of_atoms, qtb_nfreq2,
    qtb_random_array_0.data(), qtb_random_array_1.data(), qtb_random_array_2.data());
  GPU_CHECK_KERNEL
}

void Ensemble_NPH_QTB::get_target_temp()
{
  t_target = t_start + (t_stop - t_start) * get_delta();
}

void Ensemble_NPH_QTB::qtb_update_time_filter(const double target_temperature)
{
  if (fabs(target_temperature - qtb_last_filter_temperature) < 1.0e-12)
    return;

  std::vector<double> omega_H(qtb_nfreq2, 0.0);
  for (int k = 0; k < qtb_nfreq2; ++k) {
    const double k_shift = k - qtb_N_f;
    const double f_k = k_shift / (qtb_nfreq2 * qtb_h_timestep);
    if (k == qtb_N_f) { omega_H[k] = sqrt(K_B * target_temperature); continue; }
    const double energy_k = 2.0 * PI * HBAR * fabs(f_k);
    const double x = energy_k / (K_B * target_temperature);
    double qfactor = 0.5;
    if (x < 200.0) qfactor += 1.0 / (exp(x) - 1.0);
    omega_H[k] = sqrt(energy_k * qfactor);
    const double num = sin(k_shift * PI / (2.0 * qtb_alpha * qtb_N_f));
    const double den = sin(k_shift * PI / (2.0 * qtb_N_f));
    omega_H[k] *= qtb_alpha * num / den;
  }

  for (int n = 0; n < qtb_nfreq2; ++n) {
    double value = 0.0;
    const double t_n = n - qtb_N_f;
    for (int k = 0; k < qtb_nfreq2; ++k) {
      const double omega_k = (k - qtb_N_f) * PI / qtb_N_f;
      value += omega_H[k] * cos(omega_k * t_n);
    }
    qtb_time_H_host[n] = value / qtb_nfreq2;
  }
  qtb_time_H_device.copy_from_host(qtb_time_H_host.data());
  qtb_last_filter_temperature = target_temperature;
}

void Ensemble_NPH_QTB::qtb_refresh_colored_random_force()
{
  const double g3p = sqrt(2.0 * qtb_fric_coef * 12.0 / qtb_h_timestep);
  gpu_refresh_qtb_random_force<<<(qtb_number_of_atoms - 1) / 128 + 1, 128>>>(
    qtb_curand_states.data(), qtb_number_of_atoms, qtb_nfreq2,
    qtb_time_H_device.data(), g3p, atom->mass.data(),
    qtb_random_array_0.data(), qtb_random_array_1.data(), qtb_random_array_2.data(),
    qtb_fran.data(), qtb_fran.data() + qtb_number_of_atoms,
    qtb_fran.data() + qtb_number_of_atoms * 2);
  GPU_CHECK_KERNEL
}

void Ensemble_NPH_QTB::qtb_apply_half_step()
{
  const int N = qtb_number_of_atoms;
  const double dt_half = 0.5 * qtb_dt;

  gpu_apply_qtb_half_step<<<(N - 1) / 128 + 1, 128>>>(
    N, dt_half, qtb_fric_coef, atom->mass.data(),
    qtb_fran.data(), qtb_fran.data() + N, qtb_fran.data() + 2 * N,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + N,
    atom->velocity_per_atom.data() + 2 * N);
  GPU_CHECK_KERNEL

  gpu_find_momentum<<<4, 1024>>>(
    N, atom->mass.data(),
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + N,
    atom->velocity_per_atom.data() + 2 * N);
  GPU_CHECK_KERNEL

  gpu_correct_momentum<<<(N - 1) / 128 + 1, 128>>>(
    N, atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + N,
    atom->velocity_per_atom.data() + 2 * N);
  GPU_CHECK_KERNEL
}

/* PLACEHOLDER_COMPUTE */

// Integration scheme:
// compute1: press_chain -> QTB_half_kick -> barostat_v -> verlet_v -> box -> verlet_x -> box
// compute2: verlet_v -> barostat_v -> omega_dot -> QTB_half_kick -> press_chain -> thermo

void Ensemble_NPH_QTB::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0) {
    init_mttk();
  }

  // 1. Pressure chain thermostat (for barostat DOF)
  nhc_press_integrate();

  // 2. QTB thermostat half-kick (replaces nhc_temp_integrate)
  get_target_temp();
  if (qtb_counter_mu == 0) {
    qtb_update_time_filter(t_target);
    qtb_refresh_colored_random_force();
  }
  qtb_apply_half_step();

  // 3. Barostat: update omega_dot and scale velocities
  get_h_matrix_from_box();
  get_target_pressure();
  nh_omega_dot();
  nh_v_press();

  // 4. Velocity Verlet half-step (velocity)
  velocity_verlet_v();

  // 5. Propagate box
  propagate_box();

  // 6. Velocity Verlet (position)
  velocity_verlet_x();

  // 7. Propagate box again
  propagate_box();
}

void Ensemble_NPH_QTB::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  // 1. Velocity Verlet half-step (velocity)
  velocity_verlet_v();

  // 2. Barostat: scale velocities and update omega_dot
  get_h_matrix_from_box();
  nh_v_press();
  nh_omega_dot();

  // 3. QTB thermostat half-kick (replaces nhc_temp_integrate)
  qtb_apply_half_step();

  // 4. Pressure chain thermostat
  nhc_press_integrate();

  // 5. Compute thermodynamic quantities
  find_thermo();

  // 6. Update QTB counter
  qtb_counter_mu = (qtb_counter_mu + 1) % qtb_alpha;
}
