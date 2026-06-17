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

#include "ensemble_ti_superionic.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstdlib>
#include <cstring>
#include <math.h>

namespace
{

bool is_superionic_keyword(const char* token)
{
  return strcmp(token, "temp") == 0 || strcmp(token, "tperiod") == 0 ||
         strcmp(token, "tequil") == 0 || strcmp(token, "tswitch") == 0 ||
         strcmp(token, "press") == 0 || strcmp(token, "spring") == 0 ||
         strcmp(token, "uf") == 0;
}

int stage_number(SuperionicStage stage)
{
  return static_cast<int>(stage);
}

const char* csv_filename(SuperionicStage stage)
{
  return stage == SuperionicStage::stage1 ? "ti_superionic_stage1.csv"
                                          : "ti_superionic_stage2.csv";
}

const char* yaml_filename(SuperionicStage stage)
{
  return stage == SuperionicStage::stage1 ? "ti_superionic_stage1.yaml"
                                          : "ti_superionic_stage2.yaml";
}

void write_uf_pair(FILE* yaml_file, const SuperionicUFPair& pair)
{
  fprintf(yaml_file, "  - element_i: %s\n", pair.element_i.c_str());
  fprintf(yaml_file, "    element_j: %s\n", pair.element_j.c_str());
  fprintf(yaml_file, "    p: %.17g\n", pair.p);
  fprintf(yaml_file, "    sigma: %.17g\n", pair.sigma);
}

} // namespace

Ensemble_TI_Superionic::Ensemble_TI_Superionic(
  const char** params, int num_params, SuperionicStage input_stage)
  : stage(input_stage)
{
  atom = nullptr;
  box = nullptr;
  group = nullptr;
  thermo = nullptr;
  current_step = nullptr;
  total_steps = nullptr;

  temperature_coupling = 100;
  int i = 2;
  while (i < num_params) {
    if (strcmp(params[i], "temp") == 0) {
      if (i + 1 >= num_params || !is_valid_real(params[i + 1], &temperature))
        PRINT_INPUT_ERROR("Wrong inputs for temp keyword.");
      has_temperature = true;
      i += 2;
    } else if (strcmp(params[i], "tperiod") == 0) {
      if (i + 1 >= num_params || !is_valid_real(params[i + 1], &temperature_coupling))
        PRINT_INPUT_ERROR("Wrong inputs for t_period keyword.");
      i += 2;
    } else if (strcmp(params[i], "tequil") == 0) {
      if (i + 1 >= num_params || !is_valid_int(params[i + 1], &t_equil))
        PRINT_INPUT_ERROR("Wrong inputs for t_equil keyword.");
      i += 2;
    } else if (strcmp(params[i], "tswitch") == 0) {
      if (i + 1 >= num_params || !is_valid_int(params[i + 1], &t_switch))
        PRINT_INPUT_ERROR("Wrong inputs for t_switch keyword.");
      i += 2;
    } else if (strcmp(params[i], "press") == 0) {
      if (i + 1 >= num_params || !is_valid_real(params[i + 1], &target_pressure))
        PRINT_INPUT_ERROR("Wrong inputs for press keyword.");
      target_pressure /= PRESSURE_UNIT_CONVERSION;
      i += 2;
    } else if (strcmp(params[i], "spring") == 0) {
      i++;
      if (i >= num_params || is_superionic_keyword(params[i]))
        PRINT_INPUT_ERROR("Missing inputs for spring keyword.");

      if (strcmp(params[i], "auto") == 0) {
        auto_k = true;
        i++;
        while (i < num_params && !is_superionic_keyword(params[i])) {
          auto_spring_species.push_back(params[i]);
          i++;
        }
      } else {
        auto_k = false;
        while (i < num_params && !is_superionic_keyword(params[i])) {
          double k = 0.0;
          if (i + 1 >= num_params || is_superionic_keyword(params[i + 1]) ||
              !is_valid_real(params[i + 1], &k))
            PRINT_INPUT_ERROR("Wrong inputs for spring keyword.");
          spring_map[params[i]] = k;
          i += 2;
        }
      }
    } else if (strcmp(params[i], "uf") == 0) {
      if (i + 4 >= num_params)
        PRINT_INPUT_ERROR("Wrong inputs for uf keyword.");
      SuperionicUFPair pair;
      pair.element_i = params[i + 1];
      pair.element_j = params[i + 2];
      if (!is_valid_real(params[i + 3], &pair.p) || !is_valid_real(params[i + 4], &pair.sigma))
        PRINT_INPUT_ERROR("Wrong inputs for uf keyword.");
      uf_pairs.push_back(pair);
      i += 5;
    } else {
      PRINT_INPUT_ERROR("Unknown keyword.");
    }
  }

  if (t_equil < 0 || t_switch < 0)
    PRINT_INPUT_ERROR("Please specify both t_equil and t_switch.");
  if (!has_temperature)
    PRINT_INPUT_ERROR("Please specify temp.");
  if (temperature <= 0)
    PRINT_INPUT_ERROR("Temperature should > 0.");
  if (temperature_coupling < 1)
    PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
  if (t_switch <= 0)
    PRINT_INPUT_ERROR("t_switch should be > 0.");
  if (spring_map.empty() && auto_spring_species.empty())
    PRINT_INPUT_ERROR("Please specify at least one spring species.");
  if (uf_pairs.empty())
    PRINT_INPUT_ERROR("Please specify at least one uf pair.");

  printf(
    "Thermostat: target temperature is %f k, t_period is %f timesteps.\n",
    temperature,
    temperature_coupling);
  type = 3;
  c1 = exp(-0.5 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * temperature);
}

void Ensemble_TI_Superionic::init()
{
  printf("The number of steps should be set to %d!\n", 2 * (t_equil + t_switch));
  printf(
    "Superionic TI stage %d: t_switch is %d timestep, t_equil is %d timesteps.\n",
    stage_number(stage),
    t_switch,
    t_equil);

  output_file = my_fopen(csv_filename(stage), "w");
  if (stage == SuperionicStage::stage1) {
    fprintf(output_file, "lambda,dlambda,U_einstein,U_uf_self,U_uf_cross,dHdlambda\n");
  } else {
    fprintf(
      output_file,
      "lambda,dlambda,U_target,U_einstein,U_uf_self,U_uf_cross,U_aux,dHdlambda\n");
  }

  int N = atom->number_of_atoms;
  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N, rand());
  GPU_CHECK_KERNEL

  thermo_cpu.resize(thermo->size());
  initialized = true;
}

Ensemble_TI_Superionic::~Ensemble_TI_Superionic(void)
{
  int N = 0;
  if (atom != nullptr) {
    N = atom->number_of_atoms;
  }
  if (atom != nullptr && box != nullptr && N > 0) {
    V = box->get_volume() / N;
  }

  FILE* yaml_file = my_fopen(yaml_filename(stage), "w");
  fprintf(yaml_file, "stage: %d\n", stage_number(stage));
  fprintf(yaml_file, "T: %.17g\n", temperature);
  fprintf(yaml_file, "V: %.17g\n", V);
  fprintf(yaml_file, "P: %.17g\n", target_pressure);
  fprintf(yaml_file, "N_total: %d\n", N);

  if (spring_map.empty() && auto_spring_species.empty()) {
    fprintf(yaml_file, "spring_species: []\n");
  } else {
    fprintf(yaml_file, "spring_species:\n");
    for (const auto& entry : spring_map) {
      fprintf(yaml_file, "  - %s\n", entry.first.c_str());
    }
    for (const auto& element : auto_spring_species) {
      fprintf(yaml_file, "  - %s\n", element.c_str());
    }
  }

  bool wrote_self_pair = false;
  fprintf(yaml_file, "uf_self_pairs:");
  for (const auto& pair : uf_pairs) {
    if (pair.element_i == pair.element_j) {
      if (!wrote_self_pair) {
        fprintf(yaml_file, "\n");
      }
      write_uf_pair(yaml_file, pair);
      wrote_self_pair = true;
    }
  }
  if (!wrote_self_pair) {
    fprintf(yaml_file, " []\n");
  }

  bool wrote_cross_pair = false;
  fprintf(yaml_file, "uf_cross_pairs:");
  for (const auto& pair : uf_pairs) {
    if (pair.element_i != pair.element_j) {
      if (!wrote_cross_pair) {
        fprintf(yaml_file, "\n");
      }
      write_uf_pair(yaml_file, pair);
      wrote_cross_pair = true;
    }
  }
  if (!wrote_cross_pair) {
    fprintf(yaml_file, " []\n");
  }

  fprintf(yaml_file, "W_forward: %.17g\n", W_forward);
  fprintf(yaml_file, "W_backward: %.17g\n", W_backward);
  fprintf(yaml_file, "delta_F: %.17g\n", delta_F);
  fprintf(yaml_file, "F_Einstein: 0\n");
  fprintf(yaml_file, "F_UF_self: 0\n");
  fprintf(yaml_file, "F_ref: 0\n");

  if (output_file != nullptr) {
    printf("Closing %s output file...\n", csv_filename(stage));
    fclose(output_file);
  }
  fclose(yaml_file);
}

void Ensemble_TI_Superionic::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo)
{
  if (*current_step == 0 && !initialized)
    init();
  Ensemble_LAN::compute1(time_step, group, box, atoms, thermo);
}

void Ensemble_TI_Superionic::find_lambda()
{
  V = box->get_volume() / atom->number_of_atoms;
  lambda_active = false;

  const int t = *current_step - t_equil;
  const double r_switch = 1.0 / t_switch;

  if ((t >= 0) && (t <= t_switch)) {
    lambda = switch_func(t * r_switch);
    dlambda = dswitch_func(t * r_switch);
    lambda_active = true;
  } else if ((t >= t_equil + t_switch) && (t <= (t_equil + 2 * t_switch))) {
    lambda = switch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    dlambda = -dswitch_func(1.0 - (t - t_switch - t_equil) * r_switch);
    lambda_active = true;
  }

  if (lambda_active && output_file != nullptr) {
    if (stage == SuperionicStage::stage1) {
      fprintf(output_file, "%e,%e,0,0,0,0\n", lambda, dlambda);
    } else {
      fprintf(output_file, "%e,%e,0,0,0,0,0,0\n", lambda, dlambda);
    }
  }
}

void Ensemble_TI_Superionic::compute3(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atoms,
  GPU_Vector<double>& thermo,
  Force& force)
{
  (void)force;
  find_lambda();
  Ensemble_LAN::compute2(time_step, group, box, atoms, thermo);
}

double Ensemble_TI_Superionic::switch_func(double t)
{
  double t2 = t * t;
  double t5 = t2 * t2 * t;
  return ((70.0 * t2 * t2 - 315.0 * t2 * t + 540.0 * t2 - 420.0 * t + 126.0) * t5);
}

double Ensemble_TI_Superionic::dswitch_func(double t)
{
  double t2 = t * t;
  double t4 = t2 * t2;
  return ((630 * t2 * t2 - 2520 * t2 * t + 3780 * t2 - 2520 * t + 630) * t4) /
         t_switch;
}
