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

Ensemble_NH::Ensemble_NH(const char** params, int num_params)
{
  // TODO: read deviatoric stress
  // TODO: do barostat
  // parse params
  for (int i = 0; i < num_params; i++) {
    printf(params[i]);
    printf(" ");
  }
  printf("\n");

  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "temp") == 0) {
      tstat_flag = 1;
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
      pstat_flag = 1;
      if (!is_valid_real(params[i + 1], p_start))
        PRINT_INPUT_ERROR("Wrong inputs for p_start keyword.");
      p_start[1] = p_start[2] = p_start[0];
      if (!is_valid_real(params[i + 2], p_start))
        PRINT_INPUT_ERROR("Wrong inputs for p_stop keyword.");
      p_stop[1] = p_stop[2] = p_stop[0];
      if (!(is_valid_real(params[i + 3], p_period)))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      p_period[1] = p_period[2] = p_period[0];
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      // when tri, enable pstat on three off-diagonal elements, and set target stress to zero.
      if (strcmp(params[i], "tri") == 0) {
        p_period[5] = p_period[4] = p_period[3] = p_period[0];
        p_flag[5] = p_flag[4] = p_flag[3] = 1;
      }
      i += 4;
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

  const char* stress_components[6] = {"x", "y", "z", "yz", "xz", "xy"};
  if (pstat_flag) {
    for (int i = 0; i < 6; i++) {
      if (p_flag[i] == 1)
        printf(
          "%s : p_start is %f, p_stop is %f, p_period is %f timesteps\n",
          stress_components[i],
          p_start[i],
          p_stop[i],
          p_period[i]);
      else
        printf("%s will not be changed.\n", stress_components[i]);
    }
  } else
    printf("No barostat is set. Pressure is not controlled.\n");
}

Ensemble_NH::~Ensemble_NH(void) { delete[] Q, eta_dot, eta_dotdot; }

double Ensemble_NH::get_delta() { return (double)*current_step / (double)*total_steps; }

void Ensemble_NH::get_target_temp() { t_target = t_start + (t_stop - t_start) * get_delta(); }

void Ensemble_NH::velocity_verlet_step1()
{
  velocity_verlet(
    true,
    time_step,
    *group,
    atom->mass,
    atom->force_per_atom,
    atom->position_per_atom,
    atom->velocity_per_atom);
}

void Ensemble_NH::velocity_verlet_step2()
{
  velocity_verlet(
    false,
    time_step,
    *group,
    atom->mass,
    atom->force_per_atom,
    atom->position_per_atom,
    atom->velocity_per_atom);
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

void Ensemble_NH::init()
{
  // set tstat params
  // Here I negelect center of mass dof.
  tdof = atom->number_of_atoms * 3;
  dt = time_step;
  dthalf = dt / 2;
  dt4 = dt / 4;
  dt8 = dt / 8;
  t_freq = 1 / (t_period * dt);
  Q = new double[tchain];
  eta_dot = new double[tchain];
  eta_dotdot = new double[tchain];
  for (int n = 0; n < tchain; n++)
    Q[n] = eta_dot[n] = eta_dotdot[n] = 0;
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

  velocity_verlet_step1();
}

void Ensemble_NH::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet_step2();

  if (tstat_flag)
    nh_temp_integrate();

  find_thermo();
}
