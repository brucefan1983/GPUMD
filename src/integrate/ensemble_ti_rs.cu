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

#include "ensemble_ti_rs.cuh"
#include "utilities/gpu_macro.cuh"

namespace
{
static __global__ void gpu_scale_force(
  int number_of_atoms,
  double lambda,
  double* fx,
  double* fy,
  double* fz,
  double* sxx,
  double* syy,
  double* szz,
  double* sxy,
  double* sxz,
  double* syz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    fx[i] *= lambda;
    fy[i] *= lambda;
    fz[i] *= lambda;
    sxx[i] *= lambda;
    syy[i] *= lambda;
    szz[i] *= lambda;
    sxy[i] *= lambda;
    sxz[i] *= lambda;
    syz[i] *= lambda;
  }
}

} // namespace

Ensemble_TI_RS::Ensemble_TI_RS(const std::vector<std::string>& tokens)
{
  const int num_params = tokens.size();
  ensemble_type = NPT;
  int i = 2;
  while (i < num_params) {
    if (tokens[i] == "tperiod") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for tperiod keyword.");
      if (!is_valid_real(tokens[i + 1], &t_period))
        PRINT_INPUT_ERROR("Wrong inputs for p_period keyword.");
      i += 2;
    } else if (tokens[i] == "pperiod") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for pperiod keyword.");
      if (!is_valid_real(tokens[i + 1], &p_period[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
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
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      if (!is_valid_real(tokens[i + 2], &t_max))
        PRINT_INPUT_ERROR("Wrong inputs for t_max keyword.");
      t_stop = t_start;
      t_target = t_start;
      i += 3;
    } else if (
      tokens[i] == "iso" || tokens[i] == "aniso" || tokens[i] == "tri") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for pressure keyword.");
      use_barostat = true;
      if (!is_valid_real(tokens[i + 1], &p_start[0][0]))
        PRINT_INPUT_ERROR("Wrong inputs for pressure keyword.");
      p_stop[1][1] = p_stop[2][2] = p_stop[0][0] = p_start[1][1] = p_start[2][2] = p_start[0][0];
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
      i += 2;
    } else if (tokens[i] == "tswitch") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for tswitch keyword.");
      auto_switch = false;
      if (!is_valid_int(tokens[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for t_switch keyword.");
      i += 2;
    } else if (tokens[i] == "tequil") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for tequil keyword.");
      auto_switch = false;
      if (!is_valid_int(tokens[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for t_equil keyword.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Wrong input parameters.");
    }
  }
  if ((t_switch < 0 && t_equil > 0) || (t_switch > 0 && t_equil < 0)) {
    PRINT_INPUT_ERROR(
      "Error: Please specify either both t_switch and t_equil, or neither (to let the program "
      "auto-determine)");
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
  lambda_f = t_start / t_max;
  printf("Nonequilibrium thermodynamic integration:\n");
  printf("    temp_start is %f K.\n", t_start);
  printf("    temp_max is %f K.\n", t_max);
  printf("    final lambda value is %f.\n", lambda_f);
}

void Ensemble_TI_RS::init(
  const int number_of_steps, const GPU_Vector<double>& thermo)
{
  if (auto_switch) {
    t_switch = (int)(number_of_steps * 0.4);
    t_equil = (int)(number_of_steps * 0.1);
  } else
    printf("    The number of steps should be set to %d!\n", 2 * (t_switch));
  printf(
    "Nonequilibrium thermodynamic integration: t_switch is %d timestep, t_equil is %d timesteps.\n",
    t_switch,
    t_equil);
  thermo_cpu.resize(thermo.size());
  output_file = my_fopen("ti_rs.csv", "w");
  fprintf(output_file, "lambda,dlambda,enthalpy\n");
}

void Ensemble_TI_RS::find_ti_thermo(
  const Box& box,
  const std::vector<Group>& group,
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
  thermo.copy_to_host(thermo_cpu.data());
  pe = thermo_cpu[1];
}

Ensemble_TI_RS::~Ensemble_TI_RS(void)
{
  close_output_file(false);
}

void Ensemble_TI_RS::finalize_run(const Atom&, const Box&)
{
  close_output_file(true);
}

void Ensemble_TI_RS::close_output_file(const bool print_message)
{
  if (output_file != nullptr) {
    if (print_message) {
      printf("Closing ti_rs output file...\n");
    }
    fclose(output_file);
    output_file = nullptr;
  }
}

void Ensemble_TI_RS::scale_force(Atom& atom)
{
  int N = atom.number_of_atoms;
  gpu_scale_force<<<(N - 1) / 128 + 1, 128>>>(
    N,
    lambda,
    atom.force_per_atom.data(),
    atom.force_per_atom.data() + N,
    atom.force_per_atom.data() + 2 * N,
    atom.virial_per_atom.data(),
    atom.virial_per_atom.data() + N,
    atom.virial_per_atom.data() + N * 2,
    atom.virial_per_atom.data() + N * 3,
    atom.virial_per_atom.data() + N * 4,
    atom.virial_per_atom.data() + N * 5);
}

void Ensemble_TI_RS::initialize_before_first_step(
  const double time_step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  init(number_of_steps, thermo);
  Ensemble_MTTK::initialize_before_first_step(
    time_step, number_of_steps, group, box, atom, thermo);
}

void Ensemble_TI_RS::compute2(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force)
{
  find_lambda(step, box, group, atoms, thermo);
  scale_force(atoms);

  Ensemble_MTTK::compute2(
    time_step, step, number_of_steps, group, box, atoms, thermo, force);
}

void Ensemble_TI_RS::find_lambda(
  const int step,
  const Box& box,
  const std::vector<Group>& group,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  bool need_output = false;
  const double t = step - t_equil;
  const double r_switch = 1.0 / (t_switch - 1);

  if ((t >= 0) && (t < t_switch)) {
    lambda = switch_func(t * r_switch);
    dlambda = dswitch_func(t * r_switch);
    need_output = true;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    need_output = true;
  }

  if (need_output) {
    find_ti_thermo(box, group, atom, thermo);
    fprintf(
      output_file,
      "%e,%e,%e\n",
      lambda,
      dlambda,
      (pe + p_start[0][0] * box.get_volume()) / atom.number_of_atoms);
  }
}

void Ensemble_TI_RS::get_target_pressure(
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  for (int ii = 0; ii < 3; ii++) {
    p_target[ii][ii] = p_start[ii][ii] * lambda;
  }
  get_p_hydro();
  if (non_hydrostatic)
    get_sigma(step, box);
}

double Ensemble_TI_RS::switch_func(double t) { return 1 / (1 + t * (1 / lambda_f - 1)); }

double Ensemble_TI_RS::dswitch_func(double t)
{
  double a = 1 / lambda_f - 1;
  return -(a / pow((1 + a * t), 2.0)) / t_switch;
}
