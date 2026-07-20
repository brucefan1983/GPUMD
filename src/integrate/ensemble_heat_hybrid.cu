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
#include "utilities/gpu_macro.cuh"
#include <cstdlib>
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
  int type_input,
  const std::vector<int>& thermostat_type_input,
  const std::vector<int>& label_input,
  const std::vector<int>& size_input,
  const std::vector<int>& offset_input,
  double temperature_input,
  const std::vector<double>& coupling_input,
  double delta_temperature_input,
  double time_step)
{
  type = type_input;
  temperature = temperature_input;
  delta_temperature = delta_temperature_input;

  num_thermostats = thermostat_type_input.size();
  thermostat_type = thermostat_type_input;
  label = label_input;
  size = size_input;
  offset = offset_input;
  coupling = coupling_input;

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
      c1[i] = exp(-0.5 / coupling[i]);
      c2[i] = sqrt((1.0 - c1[i] * c1[i]) * K_B * target);
      curand_states[i].resize(size[i]);
      initialize_curand_states<<<(size[i] - 1) / 128 + 1, 128>>>(
        curand_states[i].data(), size[i], rand());
      GPU_CHECK_KERNEL
    }
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
  bool has_nhc = false;
  bool has_lan = false;
  for (int i = 0; i < num_thermostats; i++) {
    has_nhc = has_nhc || thermostat_type[i] == 0;
    has_lan = has_lan || thermostat_type[i] == 1;
  }

  if (has_nhc) {
    std::vector<double> ek2(number_of_groups);
    GPU_Vector<double> vcx(number_of_groups), vcy(number_of_groups), vcz(number_of_groups),
      ke(number_of_groups);
    std::vector<double> factor(num_thermostats, 1.0);
    std::vector<int> nhc_labels;
    std::vector<double> nhc_factors;

    find_vc_and_ke(group, mass, velocity_per_atom, vcx.data(), vcy.data(), vcz.data(), ke.data());
    ke.copy_to_host(ek2.data());

    for (int i = 0; i < num_thermostats; i++) {
      if (thermostat_type[i] == 0) {
        double* pos_eta = get_nhc_pos(i);
        double* vel_eta = get_nhc_vel(i);
        double* mas_eta = get_nhc_mas(i);
        double kT = K_B * target_temperature(i);
        double dN = (double)DIM * size[i];
        factor[i] = nhc(
          NOSE_HOOVER_CHAIN_LENGTH,
          pos_eta,
          vel_eta,
          mas_eta,
          ek2[label[i]],
          kT,
          dN,
          time_step * 0.5);
        energy_transferred_n[i] += ek2[label[i]] * 0.5 * (1.0 - factor[i] * factor[i]);

        nhc_labels.push_back(label[i]);
        nhc_factors.push_back(factor[i]);
      }
    }

    // Use the GPU-based scaling function
    if (!nhc_labels.empty()) {
      scale_velocity_groups(
        nhc_factors,
        nhc_labels,
        vcx.data(),
        vcy.data(),
        vcz.data(),
        ke.data(),
        group,
        velocity_per_atom);
    }
  }

  if (has_lan) {
    std::vector<double> ek2(number_of_groups);
    GPU_Vector<double> ke(number_of_groups);

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
