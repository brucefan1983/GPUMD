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
Nonequilibrium thermodynamic integration between two NEP potentials.
The first potential in run.in (lambda = 0) drives the regular force
calculation and the second potential (lambda = 1) is evaluated here.
------------------------------------------------------------------------------*/

#include "ensemble_ti_nep.cuh"
#include "utilities/gpu_macro.cuh"

namespace
{
static __global__ void gpu_add_nep2_force(
  int number_of_atoms,
  double lambda,
  double* fx,
  double* fy,
  double* fz,
  const double* fx_nep2,
  const double* fy_nep2,
  const double* fz_nep2)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_atoms) {
    fx[i] = (1 - lambda) * fx[i] + lambda * fx_nep2[i];
    fy[i] = (1 - lambda) * fy[i] + lambda * fy_nep2[i];
    fz[i] = (1 - lambda) * fz[i] + lambda * fz_nep2[i];
  }
}

static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    for (int d = 0; d < 9; ++d) {
      g_virial[n1 + d * N] = 0.0;
    }
  }
}

static __global__ void gpu_get_pe_sum(const int N, double* pe)
{
  //<<<1, 1024>>>
  int tid = threadIdx.x;
  int batch, n;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (batch = 0; batch < number_of_batches; batch++) {
    n = tid + batch * 1024;
    if (n < N)
      s_data[tid] += pe[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset)
      s_data[tid] += s_data[tid + offset];
    __syncthreads();
  }
  if (tid == 0)
    pe[0] = s_data[0];
}

} // namespace

Ensemble_TI_Nep::Ensemble_TI_Nep(const std::vector<std::string>& tokens)
{
  const int num_params = tokens.size();
  temperature_coupling = 100;
  int i = 2;
  while (i < num_params) {
    if (tokens[i] == "tswitch") {
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
    } else if (tokens[i] == "temp") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for temp keyword.");
      if (!is_valid_real(tokens[i + 1], &temperature))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      i += 2;
    } else if (tokens[i] == "tperiod") {
      if (i + 1 >= num_params)
        PRINT_INPUT_ERROR("Missing value for tperiod keyword.");
      if (!is_valid_real(tokens[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }
  if ((t_switch < 0 && t_equil >= 0) || (t_switch >= 0 && t_equil < 0)) {
    PRINT_INPUT_ERROR(
      "Error: Please specify either both t_switch and t_equil, or neither (to let the program "
      "auto-determine)");
  }
  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  type = EnsembleType::NVT_LAN;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

void Ensemble_TI_Nep::init(
  const int number_of_steps, const Atom& atom, const GPU_Vector<double>& thermo)
{
  if (auto_switch) {
    t_switch = (int)(number_of_steps * 0.4);
    t_equil = (int)(number_of_steps * 0.1);
  } else
    printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
  printf(
    "Nonequilibrium thermodynamic integration: t_switch is %d timestep, t_equil is %d timesteps.\n",
    t_switch,
    t_equil);
  output_file = my_fopen("ti_nep.csv", "w");
  fprintf(output_file, "lambda,dlambda,pe,pe_nep2\n");
  int N = atom.number_of_atoms;

  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo.size());
  potential_nep2.resize(N);
  force_nep2.resize(3 * N);
  virial_nep2.resize(9 * N);
}

void Ensemble_TI_Nep::find_thermo(
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

double Ensemble_TI_Nep::get_pe_nep2(const int number_of_atoms)
{
  // reduces in place; potential_nep2 is re-initialized every step
  double temp;
  gpu_get_pe_sum<<<1, 1024>>>(number_of_atoms, potential_nep2.data());
  GPU_CHECK_KERNEL
  potential_nep2.copy_to_host(&temp, 1);
  return temp;
}

Ensemble_TI_Nep::~Ensemble_TI_Nep(void) { close_output_file(false); }

void Ensemble_TI_Nep::finalize_run(const Atom& atom, const Box& box)
{
  FILE* yaml_file = my_fopen("ti_nep.yaml", "w");
  fprintf(yaml_file, "F_diff: %f\n", F_diff);
  fprintf(yaml_file, "T: %f\n", temperature);

  close_output_file(true);
  fclose(yaml_file);

  printf("\n");
  printf("-----------------------------------------------------------------------\n");
  printf("Helmholtz free energy difference (NEP1 - NEP2): %f eV/atom.\n", F_diff);
  printf("This value is stored in ti_nep.yaml.\n");
  printf("-----------------------------------------------------------------------\n");
}

void Ensemble_TI_Nep::close_output_file(const bool print_message)
{
  if (output_file != nullptr) {
    if (print_message) {
      printf("Closing ti_nep output file...\n");
    }
    fclose(output_file);
    output_file = nullptr;
  }
}

void Ensemble_TI_Nep::add_nep2_force(Atom& atom)
{
  int N = atom.number_of_atoms;
  gpu_add_nep2_force<<<(N - 1) / 128 + 1, 128>>>(
    N,
    lambda,
    atom.force_per_atom.data(),
    atom.force_per_atom.data() + N,
    atom.force_per_atom.data() + 2 * N,
    force_nep2.data(),
    force_nep2.data() + N,
    force_nep2.data() + 2 * N);
  GPU_CHECK_KERNEL
}

void Ensemble_TI_Nep::initialize_before_first_step(
  const double,
  const int number_of_steps,
  const std::vector<Group>&,
  Box&,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  init(number_of_steps, atom, thermo);
}

void Ensemble_TI_Nep::find_lambda(
  const int step,
  const Box& box,
  const std::vector<Group>& group,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  find_thermo(box, group, atom, thermo);
  bool need_output = false;

  const int t = step - t_equil;
  pe_nep2 = get_pe_nep2(atom.number_of_atoms);

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
    fprintf(
      output_file,
      "%e,%e,%e,%e\n",
      lambda,
      dlambda,
      pe / atom.number_of_atoms,
      pe_nep2 / atom.number_of_atoms);
    F_diff += 0.5 * (pe - pe_nep2) * fabs(dlambda) / atom.number_of_atoms;
  }
}

void Ensemble_TI_Nep::compute2(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force)
{
  if (!potentials_checked) {
    if (force.get_number_of_potentials() < 2) {
      PRINT_INPUT_ERROR("ensemble ti_nep requires two potentials (NEP1 and NEP2) in run.in.");
    }
    if (force.get_potential(1).nep_model_type == 3) {
      PRINT_INPUT_ERROR("ensemble ti_nep does not support a temperature-dependent second NEP.");
    }
    potentials_checked = true;
  }

  const int N = atoms.number_of_atoms;
  initialize_properties<<<(N - 1) / 128 + 1, 128>>>(
    N,
    force_nep2.data(),
    force_nep2.data() + N,
    force_nep2.data() + N * 2,
    potential_nep2.data(),
    virial_nep2.data());
  GPU_CHECK_KERNEL

  force.get_potential(1).compute(
    box, atoms.type, atoms.position_per_atom, potential_nep2, force_nep2, virial_nep2);

  find_lambda(step, box, group, atoms, thermo);
  add_nep2_force(atoms);

  Ensemble_LAN::compute2(time_step, step, number_of_steps, group, box, atoms, thermo, force);
}

double Ensemble_TI_Nep::switch_func(double t)
{
  double t2 = t * t;
  double t5 = t2 * t2 * t;
  return ((70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t5);
}

double Ensemble_TI_Nep::dswitch_func(double t)
{
  double t2 = t * t;
  double t4 = t2 * t2;
  return ((630 * t2 * t2 - 2520 * t2 * t + 3780 * t2 - 2520 * t + 630) * t4) / t_switch;
}
