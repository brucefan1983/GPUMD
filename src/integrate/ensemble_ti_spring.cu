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

#include "ensemble_ti_spring.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

namespace
{
static __global__ void gpu_add_spring_force(
  int number_of_atoms,
  Box box,
  double lambda,
  double* espring,
  double* k,
  double* x,
  double* y,
  double* z,
  double* x0,
  double* y0,
  double* z0,
  double* fx,
  double* fy,
  double* fz)
{
  double dx, dy, dz;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    dx = x[i] - x0[i];
    dy = y[i] - y0[i];
    dz = z[i] - z0[i];
    apply_mic(box, dx, dy, dz);
    fx[i] = (1 - lambda) * fx[i] + lambda * (-k[i] * dx);
    fy[i] = (1 - lambda) * fy[i] + lambda * (-k[i] * dy);
    fz[i] = (1 - lambda) * fz[i] + lambda * (-k[i] * dz);
    espring[i] = 0.5 * k[i] * (dx * dx + dy * dy + dz * dz);
  }
}

static __global__ void gpu_add_msd(
  int number_of_atoms,
  Box box,
  double* k,
  double* x,
  double* y,
  double* z,
  double* x0,
  double* y0,
  double* z0)
{
  double dx, dy, dz;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    dx = x[i] - x0[i];
    dy = y[i] - y0[i];
    dz = z[i] - z0[i];
    apply_mic(box, dx, dy, dz);
    k[i] += dx * dx + dy * dy + dz * dz;
  }
}

static __global__ void gpu_get_espring_sum(const int N, double* espring)
{
  //<<<1, 1024>>>
  int tid = threadIdx.x;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (patch = 0; patch < number_of_patches; patch++) {
    n = tid + patch * 1024;
    if (n < N)
      s_data[tid] += espring[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    espring[0] = s_data[0];
}

} // namespace

Ensemble_TI_Spring::Ensemble_TI_Spring(const char** params, int num_params)
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
    } else if (strcmp(params[i], "press") == 0) {
      if (!is_valid_real(params[i + 1], &target_pressure))
        PRINT_INPUT_ERROR("Wrong inputs for press keyword.");
      target_pressure /= PRESSURE_UNIT_CONVERSION;
      i += 2;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (!is_valid_real(params[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
    } else if (strcmp(params[i], "spring") == 0) {
      i++;
      auto_k = false;
      double _k;
      while (i < num_params) {
        if (!is_valid_real(params[i + 1], &_k))
          PRINT_INPUT_ERROR("Wrong inputs for k keyword.");
        spring_map[params[i]] = _k;
        i += 2;
      }
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

void Ensemble_TI_Spring::init()
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
  output_file = my_fopen("ti_spring.csv", "w");
  fprintf(output_file, "lambda,dlambda,pe,espring\n");
  int N = atom->number_of_atoms;

  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo->size());
  gpu_k.resize(N, 0);
  cpu_k.resize(N, 0);
  gpu_espring.resize(N);
  position_0.resize(3 * N);
  CHECK(gpuMemcpy(
    position_0.data(),
    atom->position_per_atom.data(),
    sizeof(double) * position_0.size(),
    gpuMemcpyDeviceToDevice));

  if (!auto_k) {
    for (int i = 0; i < N; i++) {
      std::string ele = atom->cpu_atom_symbol[i];
      if (spring_map.find(ele) == spring_map.end())
        PRINT_INPUT_ERROR("You must specify the spring constants for all the elements.");
      cpu_k[i] = spring_map[ele];
    }
    gpu_k.copy_from_host(cpu_k.data());
  }
}

void Ensemble_TI_Spring::find_thermo()
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

Ensemble_TI_Spring::~Ensemble_TI_Spring(void)
{
  double kT = K_B * temperature;
  int N = atom->number_of_atoms;
  for (int i = 0; i < N; i++) {
    cpu_k[i] = pow(cpu_k[i] / atom->cpu_mass[i], 0.5);
    cpu_k[i] = log(cpu_k[i] * HBAR / kT);
    E_Ein += cpu_k[i];
  }
  E_Ein = 3 * kT * E_Ein / N;
  V = box->get_volume() / N;

  FILE* yaml_file = my_fopen("ti_spring.yaml", "w");
  fprintf(yaml_file, "E_Einstein: %f\n", E_Ein);
  fprintf(yaml_file, "E_diff: %f\n", E_diff);
  fprintf(yaml_file, "F: %f\n", E_Ein + E_diff);
  fprintf(yaml_file, "T: %f\n", temperature);
  fprintf(yaml_file, "V: %f\n", V);
  fprintf(yaml_file, "P: %f\n", target_pressure);
  fprintf(yaml_file, "G: %f\n", E_Ein + E_diff + target_pressure * V);

  printf("Closing ti_spring output file...\n");
  fclose(output_file);
  fclose(yaml_file);

  printf("\n");
  printf("-----------------------------------------------------------------------\n");
  printf("Free energy of reference system (Einstein crystal): %f eV/atom.\n", E_Ein);
  printf("Free energy difference: %f eV/atom.\n", E_diff);
  printf(
    "Pressure: %f eV/A^3 = %f GPa.\n", target_pressure, target_pressure * PRESSURE_UNIT_CONVERSION);
  printf("Volume: %f A^3.\n", V);
  printf("Helmholtz free energy of the system of interest: %f eV/atom.\n", E_Ein + E_diff);
  printf(
    "Gibbs free energy of the system of interest: %f eV/atom.\n",
    E_Ein + E_diff + target_pressure * V);
  printf("These values are stored in ti_spring.yaml.\n");
  printf("-----------------------------------------------------------------------\n");
}

void Ensemble_TI_Spring::add_spring_force()
{
  int N = atom->number_of_atoms;
  gpu_add_spring_force<<<(N - 1) / 128 + 1, 128>>>(
    N,
    *box,
    lambda,
    gpu_espring.data(),
    gpu_k.data(),
    atom->position_per_atom.data(),
    atom->position_per_atom.data() + N,
    atom->position_per_atom.data() + 2 * N,
    position_0.data(),
    position_0.data() + N,
    position_0.data() + 2 * N,
    atom->force_per_atom.data(),
    atom->force_per_atom.data() + N,
    atom->force_per_atom.data() + 2 * N);
}

double Ensemble_TI_Spring::get_espring_sum()
{
  double temp;
  gpu_get_espring_sum<<<1, 1024>>>(atom->number_of_atoms, gpu_espring.data());
  gpu_espring.copy_to_host(&temp, 1);
  return temp;
}

void Ensemble_TI_Spring::compute1(
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

void Ensemble_TI_Spring::find_lambda()
{
  find_thermo();
  int N = atom->number_of_atoms;
  bool need_output = false;

  if (*current_step < t_equil) {
    avg_pressure += pressure / t_equil;
    if (auto_k)
      gpu_add_msd<<<(N - 1) / 128 + 1, 128>>>(
        N,
        *box,
        gpu_k.data(),
        atom->position_per_atom.data(),
        atom->position_per_atom.data() + N,
        atom->position_per_atom.data() + 2 * N,
        position_0.data(),
        position_0.data() + N,
        position_0.data() + 2 * N);
  }

  // calculate MSD and spring constants
  if ((*current_step == t_equil - 1) && auto_k) {
    gpu_k.copy_to_host(cpu_k.data());
    for (int i = 0; i < N; i++) {
      std::string ele = atom->cpu_atom_symbol[i];
      if (spring_map.find(ele) == spring_map.end())
        spring_map[ele] = 0;
      spring_map[ele] += cpu_k[i];
    }
    printf("---------------------------------------\n");
    printf("Estimating spring constants from MSD...\n");
    for (const auto& myPair : spring_map) {
      std::string ele = myPair.first;
      spring_map[ele] /= atom->number_of_type(ele) * t_equil;
      spring_map[ele] = 3 * K_B * temperature / spring_map[ele];
      printf("  %s --- %f eV/A^2\n", myPair.first.c_str(), myPair.second);
    }
    printf("---------------------------------------\n");
    for (int i = 0; i < N; i++) {
      std::string ele = atom->cpu_atom_symbol[i];
      cpu_k[i] = spring_map[ele];
    }
    gpu_k.copy_from_host(cpu_k.data());
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
    espring = get_espring_sum();
    fprintf(
      output_file,
      "%e,%e,%e,%e\n",
      lambda,
      dlambda,
      pe / atom->number_of_atoms,
      espring / atom->number_of_atoms);
    E_diff += 0.5 * (pe - espring) * abs(dlambda) / atom->number_of_atoms;
  }
}

void Ensemble_TI_Spring::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  find_lambda();
  add_spring_force();

  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
}

double Ensemble_TI_Spring::switch_func(double t)
{
  double t2 = t * t;
  double t5 = t2 * t2 * t;
  return ((70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t5);
}

double Ensemble_TI_Spring::dswitch_func(double t)
{
  double t2 = t * t;
  double t4 = t2 * t2;
  return ((630 * t2 * t2 - 2520 * t2 * t + 3780 * t2 - 2520 * t + 630) * t4) / t_switch;
}