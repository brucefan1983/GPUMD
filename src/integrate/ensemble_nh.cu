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
This integrator use Nosé-Hoover thermostat and Parrinello-Rahman barostat.
only P is set -> NPH ensemable
only T is set -> NVT ensemable
P and T are both set -> NPT ensemable
------------------------------------------------------------------------------*/

#include "ensemble_nh.cuh"

namespace
{
void print_matrix(double a[3][3])
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%f ", a[i][j]);
    }
    printf("\n ");
  }
  printf("----\n");
}

void matrix_multiply(double a[3][3], double b[3][3], double c[3][3])
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c[i][j] = 0;
      for (int k = 0; k < 3; k++)
        c[i][j] += a[i][k] * b[k][j];
    }
  }
}

__device__ void matrix_vector_multiply(double a[3][3], double b[3], double c[3])
{
  for (int i = 0; i < 3; i++) {
    c[i] = 0;
    for (int j = 0; j < 3; j++)
      c[i] += a[i][j] * b[j];
  }
}

void matrix_scale(double a[3][3], double b, double c[3][3])
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i][j] * b;
    }
  }
}

void matrix_transpose(double a[3][3], double b[3][3])
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b[i][j] = a[j][i];
    }
  }
}

void matrix_minus(double a[3][3], double b[3][3], double c[3][3])
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c[i][j] = a[i][j] - b[i][j];
    }
  }
}

} // namespace

Ensemble_NH::Ensemble_NH(const char** params, int num_params)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      h[i][j] = h_inv[i][j] = h_old[i][j] = h_old_inv[i][j] = tmp1[i][j] = tmp2[i][j] =
        sigma[i][j] = fdev[i][j] = p_start[i][j] = p_stop[i][j] = p_current[i][j] = p_target[i][j] =
          p_hydro[i][j] = p_period[i][j] = p_freq[i][j] = omega[i][j] = omega_dot[i][j] =
            omega_mass[i][j] = p_flag[i][j] = h_ref_inv[i][j] = 0;
    }
  }
  // parse params
  for (int i = 0; i < num_params; i++) {
    printf(params[i]);
    printf(" ");
  }
  printf("\n");

  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "temp") == 0) {
      tstat_flag = true;
      if (!is_valid_real(params[i + 1], &t_start))
        PRINT_INPUT_ERROR("Wrong inputs for t_start keyword.");
      if (!is_valid_real(params[i + 2], &t_stop))
        PRINT_INPUT_ERROR("Wrong inputs for t_stop keyword.");
      if (!is_valid_real(params[i + 3], &t_period))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      t_target = t_start;
      i += 4;
    } else if (
      strcmp(params[i], "iso") == 0 || strcmp(params[i], "aniso") == 0 ||
      strcmp(params[i], "tri") == 0) {
      pstat_flag = true;
      if (!is_valid_real(params[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[0][0] = p_start[1][1] = p_start[2][2] = p_start[0][0];
      if (!is_valid_real(params[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[0][0] = p_stop[1][1] = p_stop[2][2] = p_stop[0][0];
      if (!(is_valid_real(params[i + 3], &p_period[0][0])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_period[1][1] = p_period[2][2] = p_period[0][0];

      p_flag[0][0] = p_flag[1][1] = p_flag[2][2] = 1;

      if (strcmp(params[i], "iso") == 0)
        couple_type = XYZ;

      // when tri, enable pstat on three off-diagonal elements, and set target stress to zero.
      if (strcmp(params[i], "tri") == 0) {
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            if (i != j) {
              p_start[i][j] = p_start[i][j] = 0;
              p_stop[i][j] = p_stop[i][j] = 0;
              p_period[i][j] = p_period[i][j] = p_period[0][0];
              p_flag[i][j] = p_flag[i][j] = true;
            }
          }
        }
      }
      i += 4;
    } else if (strcmp(params[i], "couple") == 0) {
      if (strcmp(params[i + 1], "xyz"))
        couple_type = XYZ;
      else if (strcmp(params[i + 1], "xy"))
        couple_type = XY;
      else if (strcmp(params[i + 1], "yz"))
        couple_type = YZ;
      else if (strcmp(params[i + 1], "xz"))
        couple_type = XZ;
      else
        PRINT_INPUT_ERROR("Wrong inputs for couple keyword.");
      i += 2;
    } else if (strcmp(params[i], "x") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(params[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      if (!(is_valid_real(params[i + 3], &p_period[0][0])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_flag[0][0] = 1;
      deviatoric_flag = 1;
      pstat_flag = true;
      i += 4;
    } else if (strcmp(params[i], "y") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[1][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(params[i + 2], &p_stop[1][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      if (!(is_valid_real(params[i + 3], &p_period[1][1])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_flag[1][1] = 1;
      deviatoric_flag = 1;
      pstat_flag = true;
      i += 4;
    } else if (strcmp(params[i], "z") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[2][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(params[i + 2], &p_stop[2][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      if (!(is_valid_real(params[i + 3], &p_period[2][2])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_flag[2][2] = 1;
      deviatoric_flag = 1;
      pstat_flag = true;
      i += 4;
    } else if (strcmp(params[i], "xy") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[0][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[1][0] = p_start[0][1];
      if (!is_valid_real(params[i + 2], &p_stop[0][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[1][0] = p_stop[0][1];
      if (!(is_valid_real(params[i + 3], &p_period[0][1])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_period[1][0] = p_period[0][1];
      p_flag[1][0] = p_flag[0][1] = 1;
      deviatoric_flag = 1;
      pstat_flag = true;
      i += 4;
    } else if (strcmp(params[i], "xz") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[0][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[2][0] = p_start[0][2];
      if (!is_valid_real(params[i + 2], &p_stop[0][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[2][0] = p_stop[0][2];
      if (!(is_valid_real(params[i + 3], &p_period[0][2])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_period[2][0] = p_period[0][2];
      p_flag[2][0] = p_flag[0][2] = 1;
      deviatoric_flag = 1;
      pstat_flag = true;
      i += 4;
    } else if (strcmp(params[i], "yz") == 0) {
      if (!is_valid_real(params[i + 1], &p_start[1][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[2][1] = p_start[1][2];
      if (!is_valid_real(params[i + 2], &p_stop[1][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[2][1] = p_stop[1][2];
      if (!(is_valid_real(params[i + 3], &p_period[1][2])))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_period[2][1] = p_period[1][2];
      p_flag[2][1] = p_flag[1][2] = 1;
      deviatoric_flag = 1;
      pstat_flag = true;
      i += 4;
    } else {
      PRINT_INPUT_ERROR("Wrong inputs.");
    }
  }
  // print info summary
  printf("Use Nose-Hoover thermostat and Parrinello-Rahman barostat.\n");
  if (tstat_flag && pstat_flag)
    printf("Use NPT ensemble for this run.\n");
  else if (tstat_flag)
    printf("Use NVT ensemble for this run.\n");
  else if (pstat_flag)
    printf("Use NPH ensemble for this run.\n");
  else
    PRINT_INPUT_ERROR("No thermostat and barostat are specified in input file.");

  if (tstat_flag)
    printf(
      "Thermostat: t_start is %f, t_stop is %f, t_period is %f timesteps\n",
      t_start,
      t_stop,
      t_period);
  else
    printf("No thermostat is set. Temperature is not controlled.\n");

  const char* stress_components[3][3] = {
    {"xx", "xy", "xz"}, {"yx", "yy", "yz"}, {"zx", "zy", "zz"}};
  if (pstat_flag) {
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
  } else
    printf("No barostat is set. Pressure is not controlled.\n");
}

Ensemble_NH::~Ensemble_NH(void) { delete[] Q, eta_dot, eta_dotdot; }

void Ensemble_NH::init()
{
  // from GPa to eV/A^2
  matrix_scale(p_start, 1 / PRESSURE_UNIT_CONVERSION, p_start);
  matrix_scale(p_stop, 1 / PRESSURE_UNIT_CONVERSION, p_stop);
  // set tstat params
  // Here I negelect center of mass dof.
  tdof = atom->number_of_atoms * 3;
  dt = time_step;
  dthalf = dt / 2;
  dt4 = dt / 4;
  dt8 = dt / 8;
  dt16 = dt / 16;
  t_freq = 1 / (t_period * dt);
  Q = new double[tchain];
  eta_dot = new double[tchain];
  eta_dotdot = new double[tchain];
  for (int n = 0; n < tchain; n++)
    Q[n] = eta_dot[n] = eta_dotdot[n] = 0;

  // TODO: when NPH, there is no t_start
  int t_for_barostat = t_start;
  if (t_target < 1) {
    t_for_barostat = find_current_temperature();
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j]) {
        p_freq[i][j] = 1 / (p_period[i][j] * dt);
        omega_mass[i][j] =
          (atom->number_of_atoms + 1) * kB * t_for_barostat / (p_freq[i][j] * p_freq[i][j]);
      }
    }
  }
}

double Ensemble_NH::get_delta() { return (double)*current_step / (double)*total_steps; }

void Ensemble_NH::get_target_temp() { t_target = t_start + (t_stop - t_start) * get_delta(); }

void Ensemble_NH::get_target_pressure()
{
  double delta = get_delta();
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 3; y++) {
      p_target[x][y] = p_start[x][y] + (p_stop[x][y] - p_start[x][y]) * delta;
    }
  }
  get_p_hydro();
  if (deviatoric_flag)
    get_sigma();
}

void Ensemble_NH::get_h_matrix_from_box()
{
  box->get_inverse();
  if (box->triclinic) {
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++) {
        h[x][y] = box->cpu_h[y + x * 3];
        h_inv[x][y] = box->cpu_h[9 + y + x * 3];
      }
    }
  } else {
    for (int i = 0; i < 3; i++) {
      h[i][i] = box->cpu_h[i];
      h_inv[i][i] = box->cpu_h[9 + i];
    }
  }
}

void Ensemble_NH::copy_h_matrix_to_box()
{
  if (box->triclinic) {
    for (int x = 0; x < 3; x++) {
      for (int y = 0; y < 3; y++)
        box->cpu_h[y + x * 3] = h[x][y];
    }
  } else {
    for (int i = 0; i < 3; i++)
      box->cpu_h[i] = h[i][i];
  }
  box->get_inverse();
}

void Ensemble_NH::get_p_hydro()
{
  double hydro = 0;
  for (int i = 0; i < 3; i++)
    hydro += p_target[i][i];
  hydro /= 3;
  for (int i = 0; i < 3; i++)
    p_hydro[i][i] = hydro;
}

void Ensemble_NH::get_sigma()
{
  if (h0_reset_interval > 0) {
    if (*current_step % h0_reset_interval == 0) {
      std::copy(&h_inv[0][0], &h_inv[0][0] + 9, &h_ref_inv[0][0]);
    }
  }
  // Eq. (2.24) of Parrinello1981
  // S-p
  matrix_minus(p_target, p_hydro, tmp1);
  // h_inv * (S-p)
  matrix_multiply(h_ref_inv, tmp1, tmp2);
  matrix_transpose(h_ref_inv, tmp1);
  // h_inv * (S-p) * h_inv_T
  matrix_multiply(tmp2, tmp1, sigma);
  // h_inv * (S-p) * h_inv_T * vol
  matrix_scale(sigma, box->get_volume(), sigma);
}

void Ensemble_NH::get_deviatoric()
{
  // Eq. (1) of Shinoda2004
  matrix_multiply(h, sigma, tmp1);
  matrix_transpose(h, tmp2);
  matrix_multiply(tmp1, tmp2, fdev);
}

void Ensemble_NH::couple()
{
  double xx = p_current[0][0], yy = p_current[1][1], zz = p_current[2][2];

  if (couple_type == XYZ)
    p_current[0][0] = p_current[1][1] = p_current[2][2] = (xx + yy + zz) / 3;
  else if (couple_type == XY)
    p_current[0][0] = p_current[1][1] = (xx + yy) / 2;
  else if (couple_type == YZ)
    p_current[1][1] = p_current[2][2] = (yy + zz) / 2;
  else if (couple_type == XZ)
    p_current[0][0] = p_current[2][2] = (xx + zz) / 2;
}

void Ensemble_NH::find_current_pressure()
{
  find_thermo();
  double t[8];
  thermo->copy_to_host(t, 8);
  p_current[0][0] = t[2];
  p_current[1][1] = t[3];
  p_current[2][2] = t[4];
  p_current[0][1] = p_current[1][0] = t[5];
  p_current[0][2] = p_current[2][0] = t[6];
  p_current[1][2] = p_current[2][1] = t[7];
  if (couple_type != NONE)
    couple();
}

void Ensemble_NH::nh_omega_dot()
{
  // Eq. (1) of Shinoda2004
  find_current_pressure();
  double f_omega;
  if (deviatoric_flag)
    get_deviatoric();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j]) {
        f_omega = box->get_volume() * (p_current[i][j] - p_hydro[i][j]);
        if (deviatoric_flag)
          f_omega -= fdev[i][j];
        f_omega /= omega_mass[i][j];
        omega_dot[i][j] += f_omega * dthalf;
      }
    }
  }
}

void Ensemble_NH::propagate_box()
{
  // Eq. (1) of Shinoda2004
  // save old box
  std::copy(&h[0][0], &h[0][0] + 9, &h_old[0][0]);
  std::copy(&h_inv[0][0], &h_inv[0][0] + 9, &h_old_inv[0][0]);
  // change box, according to h_dot = omega_dot * h
  propagate_box_off_diagonal();
  propagate_box_diagonal();
  propagate_box_off_diagonal();
  scale_positions();
  copy_h_matrix_to_box();
}

void Ensemble_NH::propagate_box_off_diagonal()
{
  // compute delta_h
  matrix_scale(omega_dot, dt4, tmp1);
  matrix_multiply(tmp1, h, tmp2);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (i != j && p_flag[i][j])
        h[i][j] += tmp2[i][j];
    }
  }
}
void Ensemble_NH::propagate_box_diagonal()
{
  double expfac;
  for (int i = 0; i < 3; i++) {
    expfac = exp(dthalf * omega_dot[i][i]);
    // TODO: fix point ?
    for (int j = 0; j < 3; j++) {
      h[i][j] *= expfac;
    }
  }
}

void Ensemble_NH::find_thermo()
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
}

double Ensemble_NH::find_current_temperature()
{
  find_thermo();
  double t = 0;
  thermo->copy_to_host(&t, 1);
  return t;
}

// propagate eta_dot by 1/2 step
void Ensemble_NH::nh_temp_integrate()
{
  double expfac;
  for (int n = 0; n < tchain; n++)
    Q[n] = kB * t_target / (t_freq * t_freq);
  Q[0] *= tdof;

  // propagate eta_dot by 1/4 step
  t_current = find_current_temperature();
  eta_dotdot[0] = tdof * kB * (t_current - t_target) / Q[0];
  for (int n = tchain - 1; n >= 0; n--) {
    expfac = exp(-dt8 * eta_dot[n + 1]);
    eta_dot[n] = (expfac * eta_dot[n] + eta_dotdot[n] * dt4) * expfac;
  }

  // scale velocity
  factor_eta = exp(-dthalf * eta_dot[0]);
  scale_velocity_global(factor_eta, atom->velocity_per_atom);

  // propagate eta_dot by 1/4 step
  t_current *= factor_eta * factor_eta;
  eta_dotdot[0] = tdof * kB * (t_current - t_target) / Q[0];
  eta_dot[0] = (expfac * eta_dot[0] + eta_dotdot[0] * dt4) * expfac;

  for (int n = 1; n < tchain; n++) {
    expfac = exp(-dt8 * eta_dot[n + 1]);
    eta_dotdot[n] = (Q[n - 1] * eta_dot[n - 1] * eta_dot[n - 1] - kB * t_target) / Q[n];
    eta_dot[n] = (expfac * eta_dot[n] + eta_dotdot[n] * dt4) * expfac;
  }
}

static __global__ void gpu_scale_positions(
  int number_of_atoms,
  double hax,
  double hbx,
  double hcx,
  double hay,
  double hby,
  double hcy,
  double haz,
  double hbz,
  double hcz,
  double h_old_inv_ax,
  double h_old_inv_bx,
  double h_old_inv_cx,
  double h_old_inv_ay,
  double h_old_inv_by,
  double h_old_inv_cy,
  double h_old_inv_az,
  double h_old_inv_bz,
  double h_old_inv_cz,
  double* x,
  double* y,
  double* z)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    double old_pos[3] = {x[i], y[i], z[i]};
    double frac[3], new_pos[3];
    double h_old_inv[3][3] = {
      {h_old_inv_ax, h_old_inv_bx, h_old_inv_cx},
      {h_old_inv_ay, h_old_inv_by, h_old_inv_cy},
      {h_old_inv_az, h_old_inv_bz, h_old_inv_cz}};
    double h_new[3][3] = {{hax, hbx, hcx}, {hay, hby, hcy}, {haz, hbz, hcz}};
    // fractional position
    matrix_vector_multiply(h_old_inv, old_pos, frac);
    // new position
    matrix_vector_multiply(h_new, frac, new_pos);
    x[i] = new_pos[0];
    y[i] = new_pos[1];
    z[i] = new_pos[2];
  }
}

void Ensemble_NH::scale_positions()
{
  int n = atom->number_of_atoms;
  gpu_scale_positions<<<(n - 1) / 128 + 1, 128>>>(
    atom->number_of_atoms,
    h[0][0],
    h[0][1],
    h[0][2],
    h[1][0],
    h[1][1],
    h[1][2],
    h[2][0],
    h[2][1],
    h[2][2],
    h_old_inv[0][0],
    h_old_inv[0][1],
    h_old_inv[0][2],
    h_old_inv[1][0],
    h_old_inv[1][1],
    h_old_inv[1][2],
    h_old_inv[2][0],
    h_old_inv[2][1],
    h_old_inv[2][2],
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + n,
    atom->position_per_atom.data() + 2 * n);
}

static __global__ void gpu_nh_v_press(
  int number_of_particles,
  double time_step,
  double* vx,
  double* vy,
  double* vz,
  double omega_dot_xx,
  double omega_dot_xy,
  double omega_dot_xz,
  double omega_dot_yx,
  double omega_dot_yy,
  double omega_dot_yz,
  double omega_dot_zx,
  double omega_dot_zy,
  double omega_dot_zz)
{
  double dt4 = time_step / 4;
  double dthalf = time_step / 2;
  double factor_x = exp(-dt4 * (omega_dot_xx));
  double factor_y = exp(-dt4 * (omega_dot_yy));
  double factor_z = exp(-dt4 * (omega_dot_zz));

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    vx[i] *= factor_x;
    vy[i] *= factor_y;
    vz[i] *= factor_z;

    vx[i] += -dthalf * (vy[i] * omega_dot_yx + vz[i] * omega_dot_zx);
    vy[i] += -dthalf * (vx[i] * omega_dot_xy + vz[i] * omega_dot_zy);
    vz[i] += -dthalf * (vx[i] * omega_dot_xz + vy[i] * omega_dot_yz);

    vx[i] *= factor_x;
    vy[i] *= factor_y;
    vz[i] *= factor_z;
  }
}

void Ensemble_NH::nh_v_press()
{
  int n = atom->number_of_atoms;
  gpu_nh_v_press<<<(n - 1) / 128 + 1, 128>>>(
    n,
    time_step,
    atom->velocity_per_atom.data(),
    atom->velocity_per_atom.data() + n,
    atom->velocity_per_atom.data() + 2 * n,
    omega_dot[0][0],
    omega_dot[0][1],
    omega_dot[0][2],
    omega_dot[1][0],
    omega_dot[1][1],
    omega_dot[1][2],
    omega_dot[2][0],
    omega_dot[2][1],
    omega_dot[2][2]);
}

void Ensemble_NH::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0) {
    init();
  }

  if (tstat_flag) {
    t_target = t_start + (t_stop - t_start) * get_delta();
    nh_temp_integrate();
  }

  if (pstat_flag) {
    get_h_matrix_from_box();
    get_target_pressure();
    nh_omega_dot();
    nh_v_press();
  }

  velocity_verlet_v();

  if (pstat_flag)
    propagate_box();

  velocity_verlet_x();

  if (pstat_flag)
    propagate_box();
}

void Ensemble_NH::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet_v();

  if (pstat_flag) {
    get_h_matrix_from_box();
    nh_v_press();
  }

  if (pstat_flag)
    nh_omega_dot();

  if (tstat_flag)
    nh_temp_integrate();

  // TODO: pchain

  find_thermo();
}
