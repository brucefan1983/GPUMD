/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

/*---------------------------------------------------------------------------------------------------------------------------

    Hybrid local heat baths with Nose-Hoover chain and Langevin thermostats.

    heat_hybrid can be used to perform NEMD simulations using both Nose-Hoover
    and Langevin thermostats. It can be used to analyze systems with a single heat source
    and multiple heat sinks. This functionality requires a minimum of two thermal reservoirs.

    Syntax :
    Example 1: system with a single heat source and sink
    ensemble    heat_hybrid nhc lan <T> <T_coup> <T_coup> <delta_T> <label_source> <label_sink>

    Example 2: system with a single heat source and multiple heat sinks
    ensemble    heat_hybrid lan nhc nhc <T> <T_coup> <T_coup> <T_coup> <delta_T> <label_source> <label_sink1> <label_sink2>

    Contributors: Vivekkumar Panneerselvam and Weikang (Shanghai Jiao Tong University)

------------------------------------------------------------------------------------------------------------------------------*/

#include "ensemble_heat_hybrid.cuh"
#include "langevin_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstdlib>
#include <cstring>
#define DIM 3

static double nhc(
  int M,
  double* pos_eta,
  double* vel_eta,
  double* mas_eta,
  double Ek2,
  double kT,
  double dN,
  double dt2_particle)
{
  int n_sy = 7;
  int n_respa = 4;
  const double w[7] = {
    0.784513610477560,
    0.235573213359357,
    -1.17767998417887,
    1.31518632068391,
    -1.17767998417887,
    0.235573213359357,
    0.784513610477560};

  double factor = 1.0;

  for (int n1 = 0; n1 < n_sy; n1++) {
    double dt2 = dt2_particle * w[n1] / n_respa;
    double dt4 = dt2 * 0.5;
    double dt8 = dt4 * 0.5;
    for (int n2 = 0; n2 < n_respa; n2++) {
      double G = vel_eta[M - 2] * vel_eta[M - 2] / mas_eta[M - 2] - kT;
      vel_eta[M - 1] += dt4 * G;

      for (int m = M - 2; m >= 0; m--) {
        double tmp = exp(-dt8 * vel_eta[m + 1] / mas_eta[m + 1]);
        if (m == 0) {
          G = Ek2 - dN * kT;
        } else {
          G = vel_eta[m - 1] * vel_eta[m - 1] / mas_eta[m - 1] - kT;
        }
        vel_eta[m] = tmp * (tmp * vel_eta[m] + dt4 * G);
      }

      for (int m = M - 1; m >= 0; m--) {
        pos_eta[m] += dt2 * vel_eta[m] / mas_eta[m];
      }

      double factor_local = exp(-dt2 * vel_eta[0] / mas_eta[0]);
      Ek2 *= factor_local * factor_local;
      factor *= factor_local;

      for (int m = 0; m < M - 1; m++) {
        double tmp = exp(-dt8 * vel_eta[m + 1] / mas_eta[m + 1]);
        if (m == 0) {
          G = Ek2 - dN * kT;
        } else {
          G = vel_eta[m - 1] * vel_eta[m - 1] / mas_eta[m - 1] - kT;
        }
        vel_eta[m] = tmp * (tmp * vel_eta[m] + dt4 * G);
      }

      G = vel_eta[M - 2] * vel_eta[M - 2] / mas_eta[M - 2] - kT;
      vel_eta[M - 1] += dt4 * G;
    }
  }
  return factor;
}

Ensemble_Heat_Hybrid::Ensemble_Heat_Hybrid(
  const char** param, int num_param, const std::vector<Group>& group)
{
  type = EnsembleType::HEAT_HYBRID;
  if (num_param < 9) {
    PRINT_INPUT_ERROR("ensemble heat_hybrid needs at least 7 parameters.");
  }

  num_thermostats = 0;
  while (num_thermostats + 2 < num_param) {
    const char* type_str = param[2 + num_thermostats];
    if (strcmp(type_str, "nhc") == 0) {
      thermostat_type.push_back(0);
      ++num_thermostats;
    } else if (strcmp(type_str, "lan") == 0) {
      thermostat_type.push_back(1);
      ++num_thermostats;
    } else {
      break;
    }
  }
  if (num_thermostats < 2) {
    PRINT_INPUT_ERROR("Heat-hybrid needs at least 2 thermostats.");
  }

  int idx = 2 + num_thermostats;
  if (idx >= num_param || !is_valid_real(param[idx], &temperature)) {
    PRINT_INPUT_ERROR("Temperature should be a number.");
  }
  if (temperature <= 0.0) {
    PRINT_INPUT_ERROR("Temperature should > 0.");
  }
  ++idx;

  coupling.resize(num_thermostats);
  for (int n = 0; n < num_thermostats; ++n) {
    if (idx >= num_param || !is_valid_real(param[idx], &coupling[n])) {
      PRINT_INPUT_ERROR("Heat-hybrid damping parameter should be a number.");
    }
    if (coupling[n] < 1.0) {
      PRINT_INPUT_ERROR("Heat-hybrid damping parameter should >= 1.");
    }
    ++idx;
  }

  if (idx >= num_param || !is_valid_real(param[idx], &delta_temperature)) {
    PRINT_INPUT_ERROR("Temperature difference should be a number.");
  }
  if (delta_temperature >= temperature || delta_temperature <= -temperature) {
    PRINT_INPUT_ERROR("|Temperature difference| is too large.");
  }
  ++idx;

  label.resize(num_thermostats);
  for (int n = 0; n < num_thermostats; ++n) {
    if (idx >= num_param || !is_valid_int(param[idx], &label[n])) {
      PRINT_INPUT_ERROR("Group ID for thermostat should be an integer.");
    }
    ++idx;
  }

  if (group.empty()) {
    PRINT_INPUT_ERROR("Cannot heat/cold without grouping method.");
  }
  for (int n = 0; n < num_thermostats; ++n) {
    if (label[n] < 0 || label[n] >= group[0].number) {
      PRINT_INPUT_ERROR("Group ID for heat thermostat is out of range.");
    }
    if (group[0].cpu_size[label[n]] <= 0) {
      PRINT_INPUT_ERROR("Heat thermostat group cannot be empty.");
    }
  }
  for (int i = 0; i < num_thermostats; ++i) {
    for (int j = i + 1; j < num_thermostats; ++j) {
      if (label[i] == label[j]) {
        PRINT_INPUT_ERROR("Heat thermostats must use different groups.");
      }
    }
  }

  printf("Integrate with hybrid heating and cooling for this run.\n");
  printf("    Number of thermostats: %d\n", num_thermostats);
  for (int n = 0; n < num_thermostats; ++n) {
    printf(
      "    Thermostat %d: %s, group %d, tau = %g time_step, T = %g K\n",
      n + 1,
      thermostat_type[n] == 0 ? "NHC" : "Langevin",
      label[n],
      coupling[n],
      target_temperature(n));
  }
  printf("    Average temperature: %g K\n", temperature);
  printf("    Delta T: %g K\n", delta_temperature);
  printf(
    "    Hot thermostat (T = %g K) is group %d\n",
    temperature + delta_temperature,
    label[0]);
  for (int n = 1; n < num_thermostats; ++n) {
    printf(
      "    Cold thermostat %d (T = %g K) is group %d\n",
      n,
      temperature - delta_temperature,
      label[n]);
  }
}

void Ensemble_Heat_Hybrid::initialize_run(
  const double time_step, Atom&, Box&, const std::vector<Group>& group)
{
  size.resize(num_thermostats);
  offset.resize(num_thermostats);
  for (int i = 0; i < num_thermostats; ++i) {
    size[i] = group[0].cpu_size[label[i]];
    offset[i] = group[0].cpu_size_sum[label[i]];
  }

  // Resize vectors
  c1.resize(num_thermostats);
  c2.resize(num_thermostats);
  curand_states.resize(num_thermostats);
  energy_transferred_n.resize(num_thermostats, 0.0);

  // Resize NHC arrays
  pos_nhc.resize(num_thermostats * NOSE_HOOVER_CHAIN_LENGTH);
  vel_nhc.resize(num_thermostats * NOSE_HOOVER_CHAIN_LENGTH);
  mas_nhc.resize(num_thermostats * NOSE_HOOVER_CHAIN_LENGTH);

  for (int i = 0; i < num_thermostats; i++) {
    double target = target_temperature(i);
    if (thermostat_type[i] == 0) {
      has_nhc = true;
      nhc_labels.push_back(label[i]);
      double* pos_eta = get_nhc_pos(i);
      double* vel_eta = get_nhc_vel(i);
      double* mas_eta = get_nhc_mas(i);
      double tau = time_step * coupling[i];
      for (int m = 0; m < NOSE_HOOVER_CHAIN_LENGTH; m++) {
        pos_eta[m] = 0.0;
        vel_eta[m] = (m % 2 == 0) ? 1.0 : -1.0;
        mas_eta[m] = K_B * target * tau * tau;
      }
      mas_eta[0] *= DIM * size[i];
    } else {
      has_lan = true;
      c1[i] = exp(-0.5 / coupling[i]);
      c2[i] = sqrt((1.0 - c1[i] * c1[i]) * K_B * target);
      curand_states[i].resize(size[i]);
      initialize_curand_states<<<(size[i] - 1) / 128 + 1, 128>>>(
        curand_states[i].data(), size[i], rand());
      GPU_CHECK_KERNEL
    }
  }

  initialize_group_kinetic_energy_workspace(group[0].number);
  if (has_nhc) {
    initialize_group_com_velocity_workspace(group[0].number);
    nhc_factors.resize(nhc_labels.size());
    gpu_nhc_labels.resize(nhc_labels.size());
    gpu_nhc_factors.resize(nhc_factors.size());
    gpu_nhc_labels.copy_from_host(nhc_labels.data());
  }
}

Ensemble_Heat_Hybrid::~Ensemble_Heat_Hybrid(void) {}

double Ensemble_Heat_Hybrid::target_temperature(int index) const
{
  return temperature + ((index == 0) ? delta_temperature : -delta_temperature);
}

double* Ensemble_Heat_Hybrid::get_nhc_pos(int index)
{
  return pos_nhc.data() + index * NOSE_HOOVER_CHAIN_LENGTH;
}

double* Ensemble_Heat_Hybrid::get_nhc_vel(int index)
{
  return vel_nhc.data() + index * NOSE_HOOVER_CHAIN_LENGTH;
}

double* Ensemble_Heat_Hybrid::get_nhc_mas(int index)
{
  return mas_nhc.data() + index * NOSE_HOOVER_CHAIN_LENGTH;
}

void Ensemble_Heat_Hybrid::integrate_heat_hybrid_half(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();
  const int number_of_groups = group[0].number;

  if (has_nhc) {
    std::vector<double>& ek2 = group_kinetic_energy_cpu_;
    GPU_Vector<double>& vcx = group_com_velocity_x_;
    GPU_Vector<double>& vcy = group_com_velocity_y_;
    GPU_Vector<double>& vcz = group_com_velocity_z_;
    GPU_Vector<double>& ke = group_kinetic_energy_;

    find_vc_and_ke(group, mass, velocity_per_atom, vcx.data(), vcy.data(), vcz.data(), ke.data());
    ke.copy_to_host(ek2.data());

    int nhc_index = 0;
    for (int i = 0; i < num_thermostats; i++) {
      if (thermostat_type[i] == 0) {
        double* pos_eta = get_nhc_pos(i);
        double* vel_eta = get_nhc_vel(i);
        double* mas_eta = get_nhc_mas(i);
        double kT = K_B * target_temperature(i);
        double dN = (double)DIM * size[i];
        double factor = nhc(
          NOSE_HOOVER_CHAIN_LENGTH,
          pos_eta,
          vel_eta,
          mas_eta,
          ek2[label[i]],
          kT,
          dN,
          time_step * 0.5);
        energy_transferred_n[i] += ek2[label[i]] * 0.5 * (1.0 - factor * factor);
        nhc_factors[nhc_index++] = factor;
      }
    }

    gpu_nhc_factors.copy_from_host(nhc_factors.data());
    scale_velocity_groups(
      gpu_nhc_factors,
      gpu_nhc_labels,
      vcx.data(),
      vcy.data(),
      vcz.data(),
      ke.data(),
      group,
      velocity_per_atom);
  }

  if (has_lan) {
    std::vector<double>& ek2 = group_kinetic_energy_cpu_;
    GPU_Vector<double>& ke = group_kinetic_energy_;

    find_ke<<<number_of_groups, 512>>>(
      group[0].size.data(),
      group[0].size_sum.data(),
      group[0].contents.data(),
      mass.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms,
      velocity_per_atom.data() + 2 * number_of_atoms,
      ke.data());
    GPU_CHECK_KERNEL

    ke.copy_to_host(ek2.data());
    for (int i = 0; i < num_thermostats; i++) {
      if (thermostat_type[i] == 1) {
        energy_transferred_n[i] += ek2[label[i]] * 0.5;
      }
    }

    for (int i = 0; i < num_thermostats; i++) {
      if (thermostat_type[i] == 1) {
        gpu_langevin<<<(size[i] - 1) / 128 + 1, 128>>>(
          curand_states[i].data(),
          size[i],
          offset[i],
          group[0].contents.data(),
          c1[i],
          c2[i],
          mass.data(),
          velocity_per_atom.data(),
          velocity_per_atom.data() + number_of_atoms,
          velocity_per_atom.data() + 2 * number_of_atoms);
        GPU_CHECK_KERNEL
      }
    }

    find_ke<<<number_of_groups, 512>>>(
      group[0].size.data(),
      group[0].size_sum.data(),
      group[0].contents.data(),
      mass.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms,
      velocity_per_atom.data() + 2 * number_of_atoms,
      ke.data());
    GPU_CHECK_KERNEL

    ke.copy_to_host(ek2.data());
    for (int i = 0; i < num_thermostats; i++) {
      if (thermostat_type[i] == 1) {
        energy_transferred_n[i] -= ek2[label[i]] * 0.5;
      }
    }
  }
}

void Ensemble_Heat_Hybrid::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  integrate_heat_hybrid_half(time_step, group, atom.mass, atom.velocity_per_atom);
  velocity_verlet(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
}

void Ensemble_Heat_Hybrid::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet(
    false,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
  integrate_heat_hybrid_half(time_step, group, atom.mass, atom.velocity_per_atom);
}
