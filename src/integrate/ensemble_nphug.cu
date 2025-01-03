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

#include "ensemble_nphug.cuh"
#include "utilities/gpu_macro.cuh"

namespace
{
void matrix_scale(double a[3][3], double b, double c[3][3])
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i][j] * b;
    }
  }
}
} // namespace

Ensemble_NPHug::~Ensemble_NPHug(void) {}

Ensemble_NPHug::Ensemble_NPHug(void) {}

Ensemble_NPHug::Ensemble_NPHug(const char** params, int num_params)
{
  use_thermostat = true;
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

  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "tperiod") == 0) {
      if (!is_valid_real(params[i + 1], &t_period))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      i += 2;
    } else if (strcmp(params[i], "pperiod") == 0) {
      if (!is_valid_real(params[i + 1], &p_period[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          p_period[i][j] = p_period[0][0];
        }
      }
    } else if (
      strcmp(params[i], "iso") == 0 || strcmp(params[i], "aniso") == 0 ||
      strcmp(params[i], "tri") == 0) {
      uniaxial_compress = -1;
      use_barostat = true;
      if (!is_valid_real(params[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[1][1] = p_start[2][2] = p_start[0][0];
      if (!is_valid_real(params[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[1][1] = p_stop[2][2] = p_stop[0][0];
      p_flag[0][0] = p_flag[1][1] = p_flag[2][2] = true;

      if (strcmp(params[i], "iso") == 0)
        couple_type = XYZ;

      // when tri, enable pstat on three off-diagonal elements, and set target stress to zero.
      if (strcmp(params[i], "tri") == 0) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (i != j) {
              p_start[i][j] = 0;
              p_stop[i][j] = 0;
              p_flag[i][j] = true;
              need_scale[i][j] = false;
            }
          }
        }
      }
      i += 3;
    } else if (strcmp(params[i], "x") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(params[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      uniaxial_compress = 0;
      p_flag[0][0] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (strcmp(params[i], "y") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[1][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(params[i + 2], &p_stop[1][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      uniaxial_compress = 1;
      p_flag[1][1] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (strcmp(params[i], "z") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[2][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(params[i + 2], &p_stop[2][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      uniaxial_compress = 2;
      p_flag[2][2] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (strcmp(params[i], "p0") == 0) {
      if (!is_valid_real(params[i + 1], &p0))
        PRINT_INPUT_ERROR("Invalid p0 value.");
      p0 /= PRESSURE_UNIT_CONVERSION;
      p0_given = true;
      i += 2;
    } else if (strcmp(params[i], "v0") == 0) {
      if (!is_valid_real(params[i + 1], &v0))
        PRINT_INPUT_ERROR("Invalid v0 value.");
      v0_given = true;
      i += 2;
    } else if (strcmp(params[i], "e0") == 0) {
      if (!is_valid_real(params[i + 1], &e0))
        PRINT_INPUT_ERROR("Invalid e0 value.");
      e0_given = true;
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Wrong input parameters.");
    }
  }

  // check if there are conflicts in parameters
  if (!use_barostat)
    PRINT_INPUT_ERROR("For NPHug ensemble, you must specify barostat parameters");

  // print summary
  printf("Use Nose-Hoover thermostat and Parrinello-Rahman barostat.\n");

  printf("Thermostat: temperature will automatic converge to shock Hugoniot.\n");

  const char* stress_components[3][3] = {
    {"xx", "xy", "xz"}, {"yx", "yy", "yz"}, {"zx", "zy", "zz"}};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j] == 1)
        printf(
          "%s : p_start is %f, p_stop is %f, p_period is %f timesteps\n",
          stress_components[i][j],
          p_start[i][j],
          p_stop[i][j],
          p_period[i][j]);
      else
        printf("%s will not be changed.\n", stress_components[i][j]);
    }
  }
}

void Ensemble_NPHug::init_mttk()
{
  // from GPa to eV/A^2
  matrix_scale(p_start, 1 / PRESSURE_UNIT_CONVERSION, p_start);
  matrix_scale(p_stop, 1 / PRESSURE_UNIT_CONVERSION, p_stop);
  // set tstat params
  // Here I negelect center of mass dof.
  temperature_dof = atom->number_of_atoms * 3;
  dt = time_step;
  dt2 = dt / 2;
  dt4 = dt / 4;
  dt8 = dt / 8;
  dt16 = dt / 16;
  t_freq = 1 / (t_period * dt);
  Q = new double[tchain];
  eta_dot = new double[tchain + 1];
  eta_dotdot = new double[tchain];
  Q_p = new double[pchain];
  eta_p_dot = new double[pchain + 1];
  eta_p_dotdot = new double[pchain];

  for (int n = 0; n < tchain; n++)
    Q[n] = eta_dot[n] = eta_dotdot[n] = 0;

  for (int n = 0; n < pchain; n++)
    Q_p[n] = eta_p_dot[n] = eta_p_dotdot[n] = 0;

  eta_dot[tchain] = eta_p_dot[pchain] = 0;

  t_for_barostat = find_current_temperature();

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j]) {
        p_freq[i][j] = 1 / (p_period[i][j] * dt);
        if (p_freq_max < p_freq[i][j])
          p_freq_max = p_freq[i][j];
        omega_mass[i][j] =
          (atom->number_of_atoms + 1) * kB * t_for_barostat / (p_freq[i][j] * p_freq[i][j]);
      }
    }
  }

  // get initial thermo info
  get_thermo();
  if (!v0_given)
    v0 = v_current;
  if (!e0_given)
    e0 = e_current;
  if (!p0_given)
    p0 = p_nphug_current;
  printf("    NPHug V0: %g A^3, E0: %g eV, P0: %g GPa\n", v0, e0, p0 * PRESSURE_UNIT_CONVERSION);
}

void Ensemble_NPHug::get_thermo()
{
  find_thermo();
  thermo->copy_to_host(thermo_info, 8);
  v_current = box->get_volume();
  t_current = thermo_info[0];
  e_current = thermo_info[1] + 1.5 * atom->number_of_atoms * kB * t_current;
  p_current[0][0] = thermo_info[2];
  p_current[1][1] = thermo_info[3];
  p_current[2][2] = thermo_info[4];
  p_current[0][1] = p_current[1][0] = thermo_info[5];
  p_current[0][2] = p_current[2][0] = thermo_info[6];
  p_current[1][2] = p_current[2][1] = thermo_info[7];
  if (couple_type != NONE)
    couple();
  if (uniaxial_compress >= 0)
    p_nphug_current = p_current[uniaxial_compress][uniaxial_compress];
  else
    p_nphug_current = (p_current[0][0] + p_current[1][1] + p_current[2][2]) / 3.0;
}

void Ensemble_NPHug::get_target_temp()
{
  get_thermo();
  // calculate hugoniot
  dhugo = (0.5 * (p_nphug_current + p0) * (v0 - v_current)) + e0 - e_current;
  dhugo /= 3 * atom->number_of_atoms * kB;
  if (*current_step == 0 || *current_step % (*total_steps / 10) == 0) {
    printf("    NPHug info: current T: %f K, dHugoniot: %f K\n", t_current, dhugo);
  }
  t_target = t_current + dhugo;
}