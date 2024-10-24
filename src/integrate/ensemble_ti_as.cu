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

#include "ensemble_ti_as.cuh"
#include "utilities/gpu_macro.cuh"

Ensemble_TI_AS::Ensemble_TI_AS(const char** params, int num_params)
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

  ensemble_type = NPT;
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
    } else if (strcmp(params[i], "temp") == 0) {
      use_thermostat = true;
      if (!is_valid_real(params[i + 1], &t_start))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      t_stop = t_start;
      t_target = t_start;
      i += 2;
    } else if (
      strcmp(params[i], "iso") == 0 || strcmp(params[i], "aniso") == 0 ||
      strcmp(params[i], "tri") == 0) {
      use_barostat = true;
      if (!is_valid_real(params[i + 1], &p_min))
        PRINT_INPUT_ERROR("Wrong inputs for pressure keyword.");
      if (!is_valid_real(params[i + 2], &p_max))
        PRINT_INPUT_ERROR("Wrong inputs for pressure keyword.");
      p_stop[1][1] = p_stop[2][2] = p_stop[0][0] = p_start[1][1] = p_start[2][2] = p_start[0][0] =
        p_min;
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
    } else if (strcmp(params[i], "tswitch") == 0) {
      auto_switch = false;
      if (!is_valid_int(params[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for t_switch keyword.");
      i += 2;
    } else if (strcmp(params[i], "tequil") == 0) {
      auto_switch = false;
      if (!is_valid_int(params[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for t_equil keyword.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Wrong input parameters.");
    }
  }

  // print summary
  if (!(use_barostat && use_thermostat))
    PRINT_INPUT_ERROR("For NPT ensemble, you need to specify thermostat and barostat parameters");
  printf("Use Nose-Hoover thermostat and Parrinello-Rahman barostat.\n");
  printf("Use NPT ensemble for this run.\n");
  printf(
    "Thermostat: t_start is %f, t_stop is %f, t_period is %f timesteps\n",
    t_start,
    t_stop,
    t_period);

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
  printf("Nonequilibrium thermodynamic integration:\n");
  printf("    p_min is %f GPa.\n", p_min);
  printf("    p_max is %f GPa.\n", p_max);
  p_min /= PRESSURE_UNIT_CONVERSION;
  p_max /= PRESSURE_UNIT_CONVERSION;
}

void Ensemble_TI_AS::init()
{
  if (auto_switch) {
    t_switch = (int)(*total_steps * 0.4);
    t_equil = (int)(*total_steps * 0.1);
  } else
    printf("    The number of steps should be set to %d!\n", 2 * (t_switch));
  printf(
    "Nonequilibrium thermodynamic integration: t_switch is %d timestep, t_equil is %d timesteps.\n",
    t_switch,
    t_equil);
  thermo_cpu.resize(thermo->size());
  output_file = my_fopen("ti_as.csv", "w");
  fprintf(output_file, "p,V\n");
}

void Ensemble_TI_AS::find_thermo()
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
  pressure = (thermo_cpu[2] + thermo_cpu[3] + thermo_cpu[4]) / 3;
}

Ensemble_TI_AS::~Ensemble_TI_AS(void)
{
  printf("Closing ti_as output file...\n");
  fclose(output_file);
}

void Ensemble_TI_AS::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0)
    init();
  Ensemble_MTTK::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_AS::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  Ensemble_MTTK::compute2(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_AS::get_target_pressure()
{
  bool need_output = false;
  const int t = *current_step;
  const double r_switch = 1.0 / (t_switch - 1);
  double pp;
  double delta_p = p_max - p_min;

  if ((t >= 0) && (t < t_switch)) {
    pp = (t * r_switch) * delta_p + p_min;
    for (int ii = 0; ii < 3; ii++)
      p_target[ii][ii] = pp;
    need_output = true;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    pp = p_max - (t - t_switch) * r_switch * delta_p;
    for (int ii = 0; ii < 3; ii++)
      p_target[ii][ii] = pp;
    need_output = true;
  }

  get_p_hydro();
  if (non_hydrostatic)
    get_sigma();

  if (need_output) {
    find_thermo();
    fprintf(output_file, "%e,%e\n", pp, box->get_volume() / atom->number_of_atoms);
  }
}