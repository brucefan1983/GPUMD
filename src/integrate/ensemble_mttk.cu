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
This integrator use Nosé-Hoover thermostat and Parrinello-Rahman barostat.
only P is set -> NPH ensemable
only T is set -> NVT ensemable
P and T are both set -> NPT ensemable
------------------------------------------------------------------------------*/

#include "ensemble_mttk.cuh"
#include "utilities/gpu_macro.cuh"

namespace
{

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

Ensemble_MTTK::Ensemble_MTTK(void)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      h[i][j] = h_inv[i][j] = h_old[i][j] = h_old_inv[i][j] = tmp1[i][j] = tmp2[i][j] =
        sigma[i][j] = f_deviatoric[i][j] = p_start[i][j] = p_stop[i][j] = p_current[i][j] =
          p_target[i][j] = p_hydro[i][j] = p_freq[i][j] = omega_dot[i][j] = omega_mass[i][j] =
            p_flag[i][j] = h_ref_inv[i][j] = 0;
      p_period[i][j] = 1000;
      // TODO: if non-periodic...?
      need_scale[i][j] = true;
    }
  }
}

Ensemble_MTTK::Ensemble_MTTK(const std::vector<std::string>& tokens) : Ensemble_MTTK()
{
  const int num_params = tokens.size();
  int i = 1;
  while (i < num_params) {
    if (tokens[i] == "nvt_mttk") {
      ensemble_type = NVT;
      i += 1;
    } else if (tokens[i] == "npt_mttk") {
      ensemble_type = NPT;
      i += 1;
    } else if (tokens[i] == "nph_mttk") {
      ensemble_type = NPH;
      i += 1;
    } else if (tokens[i] == "tperiod") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for tperiod keyword.");
      if (!is_valid_real(tokens[i + 1], &t_period))
        PRINT_INPUT_ERROR("Wrong inputs for tperiod keyword.");
      i += 2;
    } else if (tokens[i] == "pperiod") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for pperiod keyword.");
      if (!is_valid_real(tokens[i + 1], &p_period[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for pperiod keyword.");
      if (p_period[0][0] < 200) {
        PRINT_INPUT_ERROR("pperiod should >= 200 timestep."); 
      }
      i += 2;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          p_period[i][j] = p_period[0][0];
        }
      }
    } else if (tokens[i] == "temp") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword temp requires two values.");
      use_thermostat = true;
      if (!is_valid_real(tokens[i + 1], &t_start))
        PRINT_INPUT_ERROR("Wrong inputs for t_start keyword.");
      if (!is_valid_real(tokens[i + 2], &t_stop))
        PRINT_INPUT_ERROR("Wrong inputs for t_stop keyword.");
      t_target = t_start;
      i += 3;
    } else if (
      tokens[i] == "iso" || tokens[i] == "aniso" || tokens[i] == "tri") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Pressure keyword requires two values.");
      use_barostat = true;
      if (!is_valid_real(tokens[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[1][1] = p_start[2][2] = p_start[0][0];
      if (!is_valid_real(tokens[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[1][1] = p_stop[2][2] = p_stop[0][0];
      p_flag[0][0] = p_flag[1][1] = p_flag[2][2] = true;

      if (tokens[i] == "iso")
        couple_type = XYZ;

      // when tri, enable pstat on three off-diagonal elements, and set target stress to zero.
      if (tokens[i] == "tri") {
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
    } else if (tokens[i] == "couple") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for couple keyword.");
      if (tokens[i + 1] == "xyz")
        couple_type = XYZ;
      else if (tokens[i + 1] == "xy")
        couple_type = XY;
      else if (tokens[i + 1] == "yz")
        couple_type = YZ;
      else if (tokens[i + 1] == "xz")
        couple_type = XZ;
      else
        PRINT_INPUT_ERROR("Wrong inputs for couple keyword.");
      i += 2;
    } else if (tokens[i] == "x") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword x requires two values.");
      if (!is_valid_real(tokens[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(tokens[i + 2], &p_stop[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_flag[0][0] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (tokens[i] == "y") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword y requires two values.");
      if (!is_valid_real(tokens[i + 1], &p_start[1][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(tokens[i + 2], &p_stop[1][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_flag[1][1] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (tokens[i] == "z") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword z requires two values.");
      if (!is_valid_real(tokens[i + 1], &p_start[2][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      if (!is_valid_real(tokens[i + 2], &p_stop[2][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_flag[2][2] = 1;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (tokens[i] == "xy") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword xy requires two values.");
      if (!is_valid_real(tokens[i + 1], &p_start[0][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[1][0] = p_start[0][1];
      if (!is_valid_real(tokens[i + 2], &p_stop[0][1]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[1][0] = p_stop[0][1];
      p_flag[1][0] = p_flag[0][1] = 1;
      need_scale[1][0] = need_scale[0][1] = false;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (tokens[i] == "xz") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword xz requires two values.");
      if (!is_valid_real(tokens[i + 1], &p_start[0][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[2][0] = p_start[0][2];
      if (!is_valid_real(tokens[i + 2], &p_stop[0][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[2][0] = p_stop[0][2];
      p_flag[2][0] = p_flag[0][2] = 1;
      need_scale[2][0] = need_scale[0][2] = false;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else if (tokens[i] == "yz") {
      if (i + 2 >= num_params)
        PRINT_INPUT_ERROR("Keyword yz requires two values.");
      if (!is_valid_real(tokens[i + 1], &p_start[1][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[2][1] = p_start[1][2];
      if (!is_valid_real(tokens[i + 2], &p_stop[1][2]))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[2][1] = p_stop[1][2];
      p_flag[2][1] = p_flag[1][2] = 1;
      need_scale[2][1] = need_scale[1][2] = false;
      non_hydrostatic = 1;
      use_barostat = true;
      i += 3;
    } else {
      PRINT_INPUT_ERROR("Wrong input parameters.");
    }
  }

  // check if there are conflicts in parameters
  if (ensemble_type == NPT) {
    if (!(use_barostat && use_thermostat))
      PRINT_INPUT_ERROR("For NPT ensemble, you need to specify thermostat and barostat parameters");
  } else if (ensemble_type == NVT) {
    if (!(!use_barostat && use_thermostat))
      PRINT_INPUT_ERROR(
        "For NVT ensemble, you need to specify thermostat parameters but no barostat parameter.");
  } else if (ensemble_type == NPH) {
    if (!(use_barostat && !use_thermostat))
      PRINT_INPUT_ERROR(
        "For NPH ensemble, you need to specify barostat parameters but no thermostat parameter.");
  } else {
    PRINT_INPUT_ERROR("Unknown ensemble type.");
  }

  // print summary
  printf("Use Nose-Hoover thermostat and Parrinello-Rahman barostat.\n");
  if (use_thermostat && use_barostat)
    printf("Use NPT ensemble for this run.\n");
  else if (use_thermostat)
    printf("Use NVT ensemble for this run.\n");
  else if (use_barostat)
    printf("Use NPH ensemble for this run.\n");
  else
    PRINT_INPUT_ERROR("No thermostat and barostat are specified in input file.");

  if (use_thermostat)
    printf(
      "Thermostat: t_start is %f, t_stop is %f, t_period is %f timesteps\n",
      t_start,
      t_stop,
      t_period);
  else
    printf("No thermostat is set. Temperature is not controlled.\n");

  const char* stress_components[3][3] = {
    {"xx", "xy", "xz"}, {"yx", "yy", "yz"}, {"zx", "zy", "zz"}};
  if (use_barostat) {
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

void Ensemble_MTTK::initialize_nose_hoover_chains()
{
  Q.reset(new double[tchain]);
  eta_dot.reset(new double[tchain + 1]);
  eta_dotdot.reset(new double[tchain]);
  Q_p.reset(new double[pchain]);
  eta_p_dot.reset(new double[pchain + 1]);
  eta_p_dotdot.reset(new double[pchain]);

  for (int n = 0; n < tchain; n++)
    Q[n] = eta_dot[n] = eta_dotdot[n] = 0;

  for (int n = 0; n < pchain; n++)
    Q_p[n] = eta_p_dot[n] = eta_p_dotdot[n] = 0;

  eta_dot[tchain] = eta_p_dot[pchain] = 0;
}

void Ensemble_MTTK::initialize_run(
  const double time_step,
  Atom&,
  Box&,
  const std::vector<Group>&)
{
  dt = time_step;
}

void Ensemble_MTTK::init_mttk(
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  printf("MTTK initializing...\n");
  // from GPa to eV/A^2
  matrix_scale(p_start, 1 / PRESSURE_UNIT_CONVERSION, p_start);
  matrix_scale(p_stop, 1 / PRESSURE_UNIT_CONVERSION, p_stop);
  // set tstat params
  // Here I neglect center of mass dof.
  temperature_dof = atom.number_of_atoms * 3;
  dt2 = dt / 2;
  dt4 = dt / 4;
  dt8 = dt / 8;
  dt16 = dt / 16;
  t_freq = 1 / (t_period * dt);
  initialize_nose_hoover_chains();

  if (use_barostat) {
    t_for_barostat = t_start;
    if (t_target < 1)
      t_for_barostat = find_current_temperature(group, box, atom, thermo);
  }
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j]) {
        p_freq[i][j] = 1 / (p_period[i][j] * dt);
        if (p_freq_max < p_freq[i][j])
          p_freq_max = p_freq[i][j];
        omega_mass[i][j] =
          (atom.number_of_atoms + 1) * kB * t_for_barostat / (p_freq[i][j] * p_freq[i][j]);
      }
    }
  }
}

double Ensemble_MTTK::get_delta(const int step, const int number_of_steps)
{
  return (double)step / (double)number_of_steps;
}

void Ensemble_MTTK::get_target_temp(
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  t_target = t_start + (t_stop - t_start) * get_delta(step, number_of_steps);
}

void Ensemble_MTTK::get_target_pressure(
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  double delta = get_delta(step, number_of_steps);
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 3; y++) {
      p_target[x][y] = p_start[x][y] + (p_stop[x][y] - p_start[x][y]) * delta;
    }
  }
  get_p_hydro();
  if (non_hydrostatic)
    get_sigma(step, box);
}

void Ensemble_MTTK::get_h_matrix_from_box(Box& box)
{
  box.get_inverse();
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 3; y++) {
      h[x][y] = box.cpu_h[y + x * 3];
      h_inv[x][y] = box.cpu_h[9 + y + x * 3];
    }
  }
}

void Ensemble_MTTK::copy_h_matrix_to_box(Box& box)
{
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 3; y++)
      box.cpu_h[y + x * 3] = h[x][y];
  }
  box.get_inverse();
}

void Ensemble_MTTK::get_p_hydro()
{
  double hydro = 0;
  for (int i = 0; i < 3; i++)
    hydro += p_target[i][i];
  hydro /= 3;
  for (int i = 0; i < 3; i++)
    p_hydro[i][i] = hydro;
}

void Ensemble_MTTK::get_sigma(const int step, const Box& box)
{
  if (h0_reset_interval > 0) {
    if (step % h0_reset_interval == 0) {
      std::copy(&h_inv[0][0], &h_inv[0][0] + 9, &h_ref_inv[0][0]);
      vol_ref = box.get_volume();
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
  matrix_scale(sigma, vol_ref, sigma);
}

void Ensemble_MTTK::get_deviatoric()
{
  // Eq. (1) of Shinoda2004
  matrix_multiply(h, sigma, tmp1);
  matrix_transpose(h, tmp2);
  matrix_multiply(tmp1, tmp2, f_deviatoric);
}

void Ensemble_MTTK::couple()
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

void Ensemble_MTTK::find_current_pressure(
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  find_thermo(group, box, atom, thermo);
  double t[8];
  thermo.copy_to_host(t, 8);
  if (use_thermostat)
    t_current = t[0];
  p_current[0][0] = t[2];
  p_current[1][1] = t[3];
  p_current[2][2] = t[4];
  p_current[0][1] = p_current[1][0] = t[5];
  p_current[0][2] = p_current[2][0] = t[6];
  p_current[1][2] = p_current[2][1] = t[7];
  if (couple_type != NONE)
    couple();
}

void Ensemble_MTTK::nh_omega_dot(
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  // Eq. (1) of Shinoda2004
  find_current_pressure(group, box, atom, thermo);
  double f_omega, V;
  V = box.get_volume();
  if (non_hydrostatic)
    get_deviatoric();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j]) {
        f_omega = V * (p_current[i][j] - p_hydro[i][j]);
        if (non_hydrostatic)
          f_omega -= f_deviatoric[i][j];
        f_omega /= omega_mass[i][j];
        omega_dot[i][j] += f_omega * dt2;
      }
    }
  }
}

void Ensemble_MTTK::propagate_box(Box& box, Atom& atom)
{
  // Eq. (1) of Shinoda2004
  // save old box
  box.get_inverse();
  get_h_matrix_from_box(box);
  std::copy(&h[0][0], &h[0][0] + 9, &h_old[0][0]);
  std::copy(&h_inv[0][0], &h_inv[0][0] + 9, &h_old_inv[0][0]);
  // change box, according to h_dot = omega_dot * h
  propagate_box_off_diagonal();
  propagate_box_diagonal();
  propagate_box_off_diagonal();
  scale_positions(atom);
  copy_h_matrix_to_box(box);
}

// void Ensemble_MTTK::propagate_box_off_diagonal()
//{
// // compute delta_h
// matrix_scale(omega_dot, dt4, tmp1);
// matrix_multiply(tmp1, h, tmp2);
// for (int i = 0; i < 3; i++) {
// for (int j = 0; j < 3; j++) {
// if (i != j && p_flag[i][j])
// h[i][j] += tmp2[i][j];
// }
// }
//}

void Ensemble_MTTK::propagate_box_off_diagonal()
{
  double expfac;
  if (p_flag[0][2]) {
    expfac = exp(dt16 * omega_dot[0][0]);
    h[0][2] *= expfac;
    h[0][2] += dt8 * (omega_dot[0][1] * h[1][2] + omega_dot[0][2] * h[2][2]);
    h[0][2] *= expfac;
  }
  if (p_flag[1][2]) {
    expfac = exp(dt8 * omega_dot[1][1]);
    h[1][2] *= expfac;
    h[1][2] += dt4 * (omega_dot[1][0] * h[0][2] + omega_dot[1][2] * h[2][2]);
    h[1][2] *= expfac;
  }
  if (p_flag[0][2]) {
    expfac = exp(dt16 * omega_dot[0][0]);
    h[0][2] *= expfac;
    h[0][2] += dt8 * (omega_dot[0][1] * h[1][2] + omega_dot[0][2] * h[2][2]);
    h[0][2] *= expfac;
  }

  if (p_flag[2][0]) {
    expfac = exp(dt16 * omega_dot[2][2]);
    h[2][0] *= expfac;
    h[2][0] += dt8 * (omega_dot[2][0] * h[0][0] + omega_dot[2][1] * h[1][0]);
    h[2][0] *= expfac;
  }
  if (p_flag[1][0]) {
    expfac = exp(dt8 * omega_dot[1][1]);
    h[1][0] *= expfac;
    h[1][0] += dt4 * (omega_dot[1][0] * h[0][0] + omega_dot[1][2] * h[2][0]);
    h[1][0] *= expfac;
  }
  if (p_flag[2][0]) {
    expfac = exp(dt16 * omega_dot[2][2]);
    h[2][0] *= expfac;
    h[2][0] += dt8 * (omega_dot[2][0] * h[0][0] + omega_dot[2][1] * h[1][0]);
    h[2][0] *= expfac;
  }

  if (p_flag[2][1]) {
    expfac = exp(dt16 * omega_dot[2][2]);
    h[2][1] *= expfac;
    h[2][1] += dt8 * (omega_dot[2][0] * h[0][1] + omega_dot[2][1] * h[1][1]);
    h[2][1] *= expfac;
  }
  if (p_flag[0][1]) {
    expfac = exp(dt8 * omega_dot[0][0]);
    h[0][1] *= expfac;
    h[0][1] += dt4 * (omega_dot[0][1] * h[1][1] + omega_dot[0][2] * h[2][1]);
    h[0][1] *= expfac;
  }
  if (p_flag[2][1]) {
    expfac = exp(dt16 * omega_dot[2][2]);
    h[2][1] *= expfac;
    h[2][1] += dt8 * (omega_dot[2][0] * h[0][1] + omega_dot[2][1] * h[1][1]);
    h[2][1] *= expfac;
  }
}

void Ensemble_MTTK::propagate_box_diagonal()
{
  // TODO: fix point ?
  double expfac;
  expfac = exp(dt4 * omega_dot[0][0]);
  h[0][0] *= expfac;
  h[0][0] += dt2 * (omega_dot[0][1] * h[1][0] + omega_dot[0][2] * h[2][0]);
  h[0][0] *= expfac;
  if (need_scale[1][0])
    h[1][0] *= expfac;
  if (need_scale[2][0])
    h[2][0] *= expfac;

  expfac = exp(dt4 * omega_dot[1][1]);
  h[1][1] *= expfac;
  h[1][1] += dt2 * (omega_dot[1][0] * h[0][1] + omega_dot[1][2] * h[2][1]);
  h[1][1] *= expfac;
  if (need_scale[0][1])
    h[0][1] *= expfac;
  if (need_scale[2][1])
    h[2][1] *= expfac;

  expfac = exp(dt4 * omega_dot[2][2]);
  h[2][2] *= expfac;
  h[2][2] += dt2 * (omega_dot[2][0] * h[0][2] + omega_dot[2][1] * h[1][2]);
  h[2][2] *= expfac;
  if (need_scale[0][2])
    h[0][2] *= expfac;
  if (need_scale[1][2])
    h[1][2] *= expfac;
}

void Ensemble_MTTK::find_thermo(
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  Ensemble::find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
}

double Ensemble_MTTK::find_current_temperature(
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (t_current_from_thermo) {
    t_current_from_thermo = false;
    return t_current;
  }
  find_thermo(group, box, atom, thermo);
  double t = 0;
  thermo.copy_to_host(&t, 1);
  return t;
}

// propagate eta_dot by 1/2 step
void Ensemble_MTTK::nhc_temp_integrate(
  const std::vector<Group>& group,
  const Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  double expfac = 0;
  for (int n = 0; n < tchain; n++)
    Q[n] = kB * t_target / (t_freq * t_freq);
  Q[0] *= temperature_dof;

  // propagate eta_dot by 1/4 step
  t_current = find_current_temperature(group, box, atom, thermo);
  eta_dotdot[0] = temperature_dof * kB * (t_current - t_target) / Q[0];
  for (int n = tchain - 1; n >= 0; n--) {
    expfac = exp(-dt8 * eta_dot[n + 1]);
    eta_dot[n] = (expfac * eta_dot[n] + eta_dotdot[n] * dt4) * expfac;
  }

  // scale velocity
  factor_eta = exp(-dt2 * eta_dot[0]);
  scale_velocity_global(factor_eta, atom.velocity_per_atom);

  // propagate eta_dot by 1/4 step
  t_current *= factor_eta * factor_eta;
  eta_dotdot[0] = temperature_dof * kB * (t_current - t_target) / Q[0];
  eta_dot[0] = (expfac * eta_dot[0] + eta_dotdot[0] * dt4) * expfac;

  for (int n = 1; n < tchain; n++) {
    expfac = exp(-dt8 * eta_dot[n + 1]);
    eta_dotdot[n] = (Q[n - 1] * eta_dot[n - 1] * eta_dot[n - 1] - kB * t_target) / Q[n];
    eta_dot[n] = (expfac * eta_dot[n] + eta_dotdot[n] * dt4) * expfac;
  }
}

void Ensemble_MTTK::nhc_press_integrate(const Atom& atom)
{

  int cell_dof; // DOF of cell
  double expfac = 0;
  double kT;
  double ke_omega_current, ke_omega_target;

  if (t_target < 1)
    kT = kB * t_for_barostat;
  else
    kT = kB * t_target;

  double nkt = (atom.number_of_atoms + 1) * kT;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j])
        omega_mass[i][j] = nkt / (p_freq[i][j] * p_freq[i][j]);
    }
  }

  Q_p[0] = kT / (p_freq_max * p_freq_max);
  for (int n = 1; n < pchain; n++)
    Q_p[n] = kT / (p_freq_max * p_freq_max);
  for (int n = 1; n < pchain; n++)
    eta_p_dotdot[n] = (Q_p[n - 1] * eta_p_dot[n - 1] * eta_p_dot[n - 1] - kT) / Q_p[n];

  ke_omega_current = 0.0;
  cell_dof = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++)
      if (i <= j && p_flag[i][j]) {
        ke_omega_current += omega_mass[i][j] * omega_dot[i][j] * omega_dot[i][j];
        cell_dof++;
      }
  }

  if (couple_type == XYZ)
    cell_dof = 1;

  ke_omega_target = cell_dof * kT;
  eta_p_dotdot[0] = (ke_omega_current - ke_omega_target) / Q_p[0];

  for (int n = pchain - 1; n >= 0; n--) {
    expfac = exp(-dt8 * eta_p_dot[n + 1]);
    eta_p_dot[n] = (eta_p_dot[n] * expfac + eta_p_dotdot[n] * dt4) * expfac;
  }

  double factor_eta_p = exp(-dt2 * eta_p_dot[0]);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j])
        omega_dot[i][j] *= factor_eta_p;
    }
  }

  ke_omega_current = 0.0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (p_flag[i][j])
        ke_omega_current += omega_mass[i][j] * omega_dot[i][j] * omega_dot[i][j];
    }
  }

  eta_p_dotdot[0] = (ke_omega_current - ke_omega_target) / Q_p[0];
  eta_p_dot[0] = (eta_p_dot[0] * expfac + eta_p_dotdot[0] * dt4) * expfac;

  for (int n = 1; n < pchain; n++) {
    expfac = exp(-dt8 * eta_p_dot[n + 1]);
    eta_p_dotdot[n] = (Q_p[n - 1] * eta_p_dot[n - 1] * eta_p_dot[n - 1] - kT) / Q_p[n];
    eta_p_dot[n] = (eta_p_dot[n] * expfac + eta_p_dotdot[n] * dt4) * expfac;
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

void Ensemble_MTTK::scale_positions(Atom& atom)
{
  int n = atom.number_of_atoms;
  gpu_scale_positions<<<(n - 1) / 128 + 1, 128>>>(
    atom.number_of_atoms,
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
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + n,
    atom.position_per_atom.data() + 2 * n);
}

static __global__ void gpu_nh_v_press(
  int number_of_particles,
  double time_step,
  double* vx,
  double* vy,
  double* vz,
  double omega_dot_ax,
  double omega_dot_bx,
  double omega_dot_cx,
  double omega_dot_ay,
  double omega_dot_by,
  double omega_dot_cy,
  double omega_dot_az,
  double omega_dot_bz,
  double omega_dot_cz)
{
  double dt4 = time_step / 4;
  double dt2 = time_step / 2;
  double factor_x = exp(-dt4 * omega_dot_ax);
  double factor_y = exp(-dt4 * omega_dot_by);
  double factor_z = exp(-dt4 * omega_dot_cz);

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    vx[i] *= factor_x;
    vy[i] *= factor_y;
    vz[i] *= factor_z;

    vx[i] += -dt2 * (vy[i] * omega_dot_bx + vz[i] * omega_dot_cx);
    vy[i] += -dt2 * (vx[i] * omega_dot_ay + vz[i] * omega_dot_cy);
    vz[i] += -dt2 * (vx[i] * omega_dot_az + vy[i] * omega_dot_bz);

    vx[i] *= factor_x;
    vy[i] *= factor_y;
    vz[i] *= factor_z;
  }
}

void Ensemble_MTTK::nh_v_press(Atom& atom)
{
  int n = atom.number_of_atoms;
  gpu_nh_v_press<<<(n - 1) / 128 + 1, 128>>>(
    n,
    dt,
    atom.velocity_per_atom.data(),
    atom.velocity_per_atom.data() + n,
    atom.velocity_per_atom.data() + 2 * n,
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

void Ensemble_MTTK::initialize_before_first_step(
  const double,
  const int,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  init_mttk(group, box, atom, thermo);
}

void Ensemble_MTTK::compute1(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (use_barostat)
    nhc_press_integrate(atom);

  if (use_thermostat) {
    get_target_temp(step, number_of_steps, group, box, atom, thermo);
    nhc_temp_integrate(group, box, atom, thermo);
  }

  if (use_barostat) {
    get_h_matrix_from_box(box);
    get_target_pressure(step, number_of_steps, group, box, atom, thermo);
    nh_omega_dot(group, box, atom, thermo);
    nh_v_press(atom);
  }

  velocity_verlet_v(dt, group, atom);

  if (use_barostat)
    propagate_box(box, atom);

  velocity_verlet_x(dt, group, atom);

  if (use_barostat)
    propagate_box(box, atom);
}

void Ensemble_MTTK::compute2(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo,
  Force& force)
{
  velocity_verlet_v(dt, group, atom);

  if (use_barostat) {
    get_h_matrix_from_box(box);
    nh_v_press(atom);
  }

  if (use_barostat)
    nh_omega_dot(group, box, atom, thermo);

  if (use_thermostat) {
    if (use_barostat)
      t_current_from_thermo = true;
    nhc_temp_integrate(group, box, atom, thermo);
  }

  if (use_barostat)
    nhc_press_integrate(atom);

  find_thermo(group, box, atom, thermo);
}
