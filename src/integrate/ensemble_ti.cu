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

#include "ensemble_ti.cuh"
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

Ensemble_TI::Ensemble_TI(const char** params, int num_params)
{
  temperature_coupling = 100;
  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "lambda") == 0) {
      if (!is_valid_real(params[i + 1], &lambda))
        PRINT_INPUT_ERROR("Wrong inputs for lambda keyword.");
      if (lambda < 0 || lambda > 1)
        PRINT_INPUT_ERROR("lambda value should be between 0 and 1.");
      i += 2;
    } else if (strcmp(params[i], "temp") == 0) {
      if (!is_valid_real(params[i + 1], &temperature))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      i += 2;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (!is_valid_real(params[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
    } else if (strcmp(params[i], "spring") == 0) {
      i++;
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
  printf("lambda is %f.\n", lambda);
  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  type = 3;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

void Ensemble_TI::init()
{
  printf("Opening TI output file...\n");
  output_file = my_fopen("ti.csv", "w");
  fprintf(output_file, "pe,espring\n");
  int N = atom->number_of_atoms;

  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo->size());
  gpu_k.resize(N);
  cpu_k.resize(N);
  for (int i = 0; i < N; i++) {
    std::string ele = atom->cpu_atom_symbol[i];
    if (spring_map.find(ele) == spring_map.end())
      PRINT_INPUT_ERROR("You must specify the spring constants for all the elements.");
    cpu_k[i] = spring_map[ele];
  }
  gpu_k.copy_from_host(cpu_k.data());
  gpu_espring.resize(N);
  position_0.resize(3 * N);
  CHECK(gpuMemcpy(
    position_0.data(),
    atom->position_per_atom.data(),
    sizeof(double) * position_0.size(),
    gpuMemcpyDeviceToDevice));
}

void Ensemble_TI::find_thermo()
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
  espring = get_espring_sum();
}

Ensemble_TI::~Ensemble_TI(void)
{
  printf("Closing TI output file...\n");
  fclose(output_file);
}

void Ensemble_TI::add_spring_force()
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

double Ensemble_TI::get_espring_sum()
{
  double temp;
  gpu_get_espring_sum<<<1, 1024>>>(atom->number_of_atoms, gpu_espring.data());
  gpu_espring.copy_to_host(&temp, 1);
  return temp;
}

void Ensemble_TI::compute1(
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

void Ensemble_TI::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  find_thermo();
  fprintf(output_file, "%e,%e\n", pe / atom->number_of_atoms, espring / atom->number_of_atoms);
  add_spring_force();

  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
}