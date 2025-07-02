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

#include "ensemble_ti_liquid.cuh"
#include "force/force.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#include <unordered_map>

namespace
{

static __global__ void
init_UF_force(int number_of_atoms, double* fx_UF, double* fy_UF, double* fz_UF)
{

  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < number_of_atoms) {
    fx_UF[i] = 0;
    fy_UF[i] = 0;
    fz_UF[i] = 0;
  }
}

static __global__ void calc_UF_force(
  int number_of_atoms,
  Box box,
  double lambda,
  double* eUF,
  double sigma_sqrd,
  double p,
  double beta,
  const int* g_NN,
  const int* g_NL,
  double* g_x,
  double* g_y,
  double* g_z,
  double* fx,
  double* fy,
  double* fz,
  double* fx_UF,
  double* fy_UF,
  double* fz_UF)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < number_of_atoms) {

    double tmp_fx_UF = 0;
    double tmp_fy_UF = 0;
    double tmp_fz_UF = 0;
    double x1 = g_x[i];
    double y1 = g_y[i];
    double z1 = g_z[i];
    double eUF_i = 0;

    for (int i1 = 0; i1 < g_NN[i]; ++i1) {

      int n2 = g_NL[i + number_of_atoms * i1];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      double x12 = double(x12double);
      double y12 = double(y12double);
      double z12 = double(z12double);
      double d12_sqrd = x12 * x12 + y12 * y12 + z12 * z12;
      double factor = -2 * p / (beta * sigma_sqrd * (exp((d12_sqrd / sigma_sqrd)) - 1));

      tmp_fx_UF = x12 * factor;
      tmp_fy_UF = y12 * factor;
      tmp_fz_UF = z12 * factor;
      eUF_i -= p / beta * logf(1 - exp(-d12_sqrd / sigma_sqrd));

      fx_UF[i] += tmp_fx_UF;
      fy_UF[i] += tmp_fy_UF;
      fz_UF[i] += tmp_fz_UF;
    }

    eUF[i] = eUF_i / 2;
  }
}

static __global__ void gpu_add_UF_force(
  int number_of_atoms,
  Box box,
  double lambda,
  double* eUF,
  double sigma_sqrd,
  double p,
  double beta,
  const int* g_NN,
  const int* g_NL,
  double* g_x,
  double* g_y,
  double* g_z,
  double* fx,
  double* fy,
  double* fz,
  double* fx_UF,
  double* fy_UF,
  double* fz_UF)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  fy[i] = (1 - lambda) * fy[i] + lambda * fy_UF[i];
  fz[i] = (1 - lambda) * fz[i] + lambda * fz_UF[i];
  fx[i] = (1 - lambda) * fx[i] + lambda * fx_UF[i];
}

static __global__ void gpu_get_UF_sum(const int N, double* eUF)
{

  int tid = threadIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N)
      s_data[tid] += eUF[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    eUF[0] = s_data[0];
}

} // namespace

Ensemble_TI_Liquid::Ensemble_TI_Liquid(const char** params, int num_params)
{
  temperature_coupling = 100;
  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "tswitch") == 0) {
      auto_switch = false;
      if (!is_valid_int(params[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for t_switch keyword.");
      i += 2;
    } else if (strcmp(params[i], "tequil") == 0) {
      auto_switch = false;
      if (!is_valid_int(params[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for t_equil keyword.");
      i += 2;
    } else if (strcmp(params[i], "temp") == 0) {
      if (!is_valid_real(params[i + 1], &temperature))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      i += 2;
      beta = 1 / (temperature * 8.6173 * 1e-5);
    } else if (strcmp(params[i], "press") == 0) {
      if (!is_valid_real(params[i + 1], &target_pressure))
        PRINT_INPUT_ERROR("Wrong inputs for press keyword.");
      target_pressure /= PRESSURE_UNIT_CONVERSION;
      i += 2;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (!is_valid_real(params[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
    } else if (strcmp(params[i], "sigmasqrd") == 0) {
      if (!is_valid_real(params[i + 1], &sigma_sqrd))
        PRINT_INPUT_ERROR("Wrong inputs for sigmasqrd keyword.");
      i += 2;
    } else if (strcmp(params[i], "p") == 0) {
      if (!is_valid_real(params[i + 1], &p))
        PRINT_INPUT_ERROR("Wrong inputs for p keyword.");

      if (p != 1 && p != 25 && p != 50 && p != 75 && p != 100)
        PRINT_INPUT_ERROR("Please pick p equal to 1, 25, 50, 75 or 100.");

      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }
  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  type = 3;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

double
Ensemble_TI_Liquid::fe(double x, const double coef[4], const double sum_spline[106], int index)
{
  double result;
  double x_0 = 0.0;

  if (x < 0.0025) {
    result = coef[0] * (x * x) / 2.0 + coef[1] * x;
    return result;
  } else if (x < 0.1) {
    if (static_cast<int>(x * 10000) % 25 == 0) {
      return sum_spline[index - 1];
    } else {
      x_0 = 0.0025 * static_cast<int>(x * 400);
    }
  } else if (x < 1) {
    if (static_cast<int>(x * 1000) % 25 == 0) {
      return sum_spline[index - 1];
    } else {
      x_0 = 0.025 * static_cast<int>(x * 40);
    }
  } else if (x < 4) {
    if (static_cast<int>(x * 100) % 10 == 0) {
      return sum_spline[index - 1];
    } else {
      x_0 = 0.1 * static_cast<int>(x * 10);
    }
  } else {
    return sum_spline[index];
  }

  result = sum_spline[index - 1] + coef[0] * (x * x - x_0 * x_0) / 2.0 + coef[1] * (x - x_0) +
           (coef[2] - 1.0) * std::log(x / x_0) - coef[3] * (1.0 / x - 1.0 / x_0);

  return result;
}

void Ensemble_TI_Liquid::init()
{
  if (auto_switch) {
    t_switch = (int)(*total_steps * 0.4);
    t_equil = (int)(*total_steps * 0.1);
  } else
    printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
  printf(
    "Nonequilibrium thermodynamic integration: t_switch is %d timestep, t_equil is %d timesteps.\n",
    t_switch,
    t_equil);
  output_file = my_fopen("ti_liquid.csv", "w");
  fprintf(output_file, "lambda,dlambda,pe,eUF\n");
  int N = atom->number_of_atoms;

  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo->size());
  gpu_eUF.resize(N);
}

void Ensemble_TI_Liquid::find_thermo()
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

  pe = thermo_cpu[1];
  pressure = (thermo_cpu[2] + thermo_cpu[3] + thermo_cpu[4]) / 3;
}

Ensemble_TI_Liquid::~Ensemble_TI_Liquid(void)
{

  double kT = K_B * temperature;
  int N = atom->number_of_atoms;
  const std::vector<double>& masses = atom->cpu_mass;
  const std::vector<int>& types = atom->cpu_type;
  std::unordered_map<int, int> species_count;

  V = box->get_volume() / N; // 1/V is the number density, rho

  double x_UF = pow(PI * sigma_sqrd, 1.5) / (2.0 * V);

  int index = 0;
  if (x_UF < 0.1) {
    index = 0 + static_cast<int>(x_UF * 400);
  } else if (x_UF < 1) {
    index = 40 + static_cast<int>(x_UF * 40 - 4);
  } else if (x_UF < 4) {
    index = 76 + static_cast<int>(x_UF * 10 - 10);
  } else {
    index = 105;
  }

  double coef[4] = {0.0};
  for (int n = 0; n < 4; n++) {
    if (p == 1) {
      coef[n] = spline1[index][n];
    } else if (p == 25) {
      coef[n] = spline25[index][n];
    } else if (p == 50) {
      coef[n] = spline50[index][n];
    } else if (p == 75) {
      coef[n] = spline75[index][n];
    } else if (p == 100) {
      coef[n] = spline100[index][n];
    }
  }

  double sum_spline[106] = {0.0};
  for (int n = 0; n < 106; n++) {
    if (p == 1) {
      sum_spline[n] = sum_spline1[n];
    } else if (p == 25) {
      sum_spline[n] = sum_spline25[n];
    } else if (p == 50) {
      sum_spline[n] = sum_spline50[n];
    } else if (p == 75) {
      sum_spline[n] = sum_spline75[n];
    } else {
      sum_spline[n] = sum_spline100[n];
    }
  }
  double F_UF = 0;

  F_UF = fe(x_UF, coef, sum_spline, index) * kT * N;

  double c_sum = 0;
  double de_broigle_sum = 0;

  for (int i = 0; i < N; ++i) {
    de_broigle_sum += log(HBAR * sqrt(2 * PI / (masses[i] * kT)));
    species_count[types[i]]++;
  }

  for (const auto& entry : species_count) {
    int count = entry.second;
    double cn = static_cast<double>(count) / N;
    
    if (cn > 0) {
      c_sum += cn * log(cn);
    }
  }

  double F_IG = N * kT * (log(1 / V) - 1 + c_sum) + 3 * kT * de_broigle_sum;
  E_ref = (F_UF + F_IG) / N;

  FILE* yaml_file = my_fopen("ti_liquid.yaml", "w");
  fprintf(yaml_file, "E_UFmodel: %f\n", E_ref);
  fprintf(yaml_file, "ES_diff: %f\n", E_diff);
  fprintf(yaml_file, "F: %f\n", E_ref + E_diff);
  fprintf(yaml_file, "T: %f\n", temperature);
  fprintf(yaml_file, "V: %f\n", V);
  fprintf(yaml_file, "P: %f\n", target_pressure);
  fprintf(yaml_file, "G: %f\n", E_ref + E_diff + target_pressure * V);

  printf("Closing ti_liquid output file...\n");
  fclose(output_file);
  fclose(yaml_file);

  printf("\n");
  printf("-----------------------------------------------------------------------\n");
  printf("Free energy of reference system (UF model): %f eV/atom.\n", E_ref);
  printf("Free energy difference: %f eV/atom.\n", E_diff);
  printf(
    "Pressure: %f eV/A^3 = %f GPa.\n", target_pressure, target_pressure * PRESSURE_UNIT_CONVERSION);
  printf("Volume: %f A^3.\n", V);
  printf("Helmholtz free energy of the system of interest: %f eV/atom.\n", E_ref + E_diff);
  printf(
    "Gibbs free energy of the system of interest: %f eV/atom.\n",
    E_ref + E_diff + target_pressure * V);
  printf("These values are stored in ti_liquid.yaml.\n");
  printf("-----------------------------------------------------------------------\n");
}

void Ensemble_TI_Liquid::add_UF_force(Force& force)
{

  int N = atom->number_of_atoms;

  const GPU_Vector<int>& NN = force.potentials[0]->get_NN_radial_ptr();

  const GPU_Vector<int>& NL = force.potentials[0]->get_NL_radial_ptr();

  GPU_Vector<double> fx_UF;
  fx_UF.resize(N);
  GPU_Vector<double> fy_UF;
  fy_UF.resize(N);
  GPU_Vector<double> fz_UF;
  fz_UF.resize(N);

  init_UF_force<<<(N - 1) / 128 + 1, 128>>>(N, fx_UF.data(), fy_UF.data(), fz_UF.data());

  calc_UF_force<<<(N - 1) / 128 + 1, 128>>>(
    N,
    *box,
    lambda,
    gpu_eUF.data(),
    sigma_sqrd,
    p,
    beta,
    NN.data(),
    NL.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    atom->force_per_atom.data(),
    atom->force_per_atom.data() + N,
    atom->force_per_atom.data() + 2 * N,
    fx_UF.data(),
    fy_UF.data(),
    fz_UF.data());

  gpu_add_UF_force<<<(N - 1) / 128 + 1, 128>>>(
    N,
    *box,
    lambda,
    gpu_eUF.data(),
    sigma_sqrd,
    p,
    beta,
    NN.data(),
    NL.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    atom->force_per_atom.data(),
    atom->force_per_atom.data() + N,
    atom->force_per_atom.data() + 2 * N,
    fx_UF.data(),
    fy_UF.data(),
    fz_UF.data());
  gpuDeviceSynchronize();
}

double Ensemble_TI_Liquid::get_UF_sum()
{
  double temp;
  gpu_get_UF_sum<<<1, 1024>>>(atom->number_of_atoms, gpu_eUF.data());
  gpu_eUF.copy_to_host(&temp, 1);
  return temp;
}

void Ensemble_TI_Liquid::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0)
    init();
  Ensemble_LAN::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Liquid::find_lambda()
{
  find_thermo();
  int N = atom->number_of_atoms;
  bool need_output = false;

  if (*current_step < t_equil) {
    avg_pressure += pressure / t_equil;
  }

  const int t = *current_step - t_equil;
  const double r_switch = 1.0 / t_switch;

  if ((t >= 0) && (t <= t_switch)) {
    lambda = switch_func(t * r_switch);
    dlambda = dswitch_func(t * r_switch);
    need_output = true;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    need_output = true;
  }

  if (need_output) {
    eUF = get_UF_sum();
    fprintf(output_file, "%e,%e,%e,%e\n", lambda, dlambda, pe / N, eUF / N);
    E_diff += 0.5 * (pe - eUF) * abs(dlambda) / N;
  }
}

void Ensemble_TI_Liquid::compute3(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force_object)
{

  find_lambda();

  add_UF_force(force_object);

  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
}

double Ensemble_TI_Liquid::switch_func(double t)
{
  double t2 = t * t;
  double t5 = t2 * t2 * t;
  return ((70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t5);
}

double Ensemble_TI_Liquid::dswitch_func(double t)
{
  double t2 = t * t;
  double t4 = t2 * t2;
  return ((630 * t2 * t2 - 2520 * t2 * t + 3780 * t2 - 2520 * t + 630) * t4) / t_switch;
}
