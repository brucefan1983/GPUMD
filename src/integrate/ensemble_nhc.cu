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
The Nose-Hoover chain thermostat
[1] M. E. Tuckerman, Statistical Mechanics: Theory and Molecular Simulation.
Oxford University Press, 2010.
------------------------------------------------------------------------------*/

#include "ensemble_nhc.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#define DIM 3

Ensemble_NHC::Ensemble_NHC(
  const char** param, int num_param, const std::vector<Group>& group)
{
  parse(param, num_param, group);
}

void Ensemble_NHC::parse_heat_groups(
  const char** param, const std::vector<Group>& group)
{
  if (!is_valid_int(param[5], &source)) {
    PRINT_INPUT_ERROR("Group ID for heat source should be an integer.");
  }
  if (!is_valid_int(param[6], &sink)) {
    PRINT_INPUT_ERROR("Group ID for heat sink should be an integer.");
  }
  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot heat/cold without grouping method.");
  }
  if (source == sink) {
    PRINT_INPUT_ERROR("Source and sink cannot be the same group.");
  }
  if (source < 0) {
    PRINT_INPUT_ERROR("Group ID for heat source should >= 0.");
  }
  if (source >= group[0].number) {
    PRINT_INPUT_ERROR("Group ID for heat source should < #groups.");
  }
  if (sink < 0) {
    PRINT_INPUT_ERROR("Group ID for heat sink should >= 0.");
  }
  if (sink >= group[0].number) {
    PRINT_INPUT_ERROR("Group ID for heat sink should < #groups.");
  }
}

void Ensemble_NHC::parse(
  const char** param, int num_param, const std::vector<Group>& group)
{
  if (strcmp(param[1], "nvt_nhc") == 0) {
    type = EnsembleType::NVT_NHC;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_nhc should have 3 parameters.");
    }

    if (!is_valid_real(param[2], &temperature1_)) {
      PRINT_INPUT_ERROR("Initial temperature should be a number.");
    }
    if (temperature1_ <= 0.0) {
      PRINT_INPUT_ERROR("Initial temperature should > 0.");
    }
    if (!is_valid_real(param[3], &temperature2_)) {
      PRINT_INPUT_ERROR("Final temperature should be a number.");
    }
    if (temperature2_ <= 0.0) {
      PRINT_INPUT_ERROR("Final temperature should > 0.");
    }
    temperature = temperature1_;
    if (!is_valid_real(param[4], &temperature_coupling)) {
      PRINT_INPUT_ERROR("Temperature coupling should be a number.");
    }
    if (temperature_coupling < 1.0) {
      PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
    }

    printf("Use NVT ensemble for this run.\n");
    printf("    choose the Nose-Hoover chain method.\n");
    printf("    initial temperature is %g K.\n", temperature1_);
    printf("    final temperature is %g K.\n", temperature2_);
    printf("    tau_T is %g time_step.\n", temperature_coupling);
    return;
  }

  if (strcmp(param[1], "heat_nhc") == 0) {
    type = EnsembleType::HEAT_NHC;
    if (num_param != 7) {
      PRINT_INPUT_ERROR("ensemble heat_nhc should have 5 parameters.");
    }
  } else if (strcmp(param[1], "heat_nhc_power") == 0) {
    type = EnsembleType::HEAT_NHC_POWER;
    if (num_param != 7) {
      PRINT_INPUT_ERROR("ensemble heat_nhc_power should have 5 parameters.");
    }
  } else {
    PRINT_INPUT_ERROR("Invalid Nose-Hoover chain ensemble type.");
  }

  if (!is_valid_real(param[2], &temperature)) {
    PRINT_INPUT_ERROR("Temperature should be a number.");
  }
  if (temperature <= 0.0) {
    PRINT_INPUT_ERROR("Temperature should > 0.");
  }
  if (!is_valid_real(param[3], &temperature_coupling)) {
    PRINT_INPUT_ERROR("Temperature coupling should be a number.");
  }
  if (temperature_coupling < 1.0) {
    PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
  }

  if (type == EnsembleType::HEAT_NHC) {
    if (!is_valid_real(param[4], &delta_temperature)) {
      PRINT_INPUT_ERROR("Temperature difference should be a number.");
    }
    if (delta_temperature >= temperature || delta_temperature <= -temperature) {
      PRINT_INPUT_ERROR("|Temperature difference| is too large.");
    }
  } else {
    if (!is_valid_real(param[4], &delta_temperature)) {
      PRINT_INPUT_ERROR("Heating power should be a number.");
    }
    if (delta_temperature <= 0.0) {
      PRINT_INPUT_ERROR("Heating power should > 0.");
    }
  }

  parse_heat_groups(param, group);

  if (type == EnsembleType::HEAT_NHC) {
    printf("Integrate with heating and cooling for this run.\n");
    printf("    choose the Nose-Hoover chain method.\n");
    printf("    average temperature is %g K.\n", temperature);
    printf("    tau_T is %g time_step.\n", temperature_coupling);
    printf("    delta_T is %g K.\n", delta_temperature);
    printf("    T_hot is %g K.\n", temperature + delta_temperature);
    printf("    T_cold is %g K.\n", temperature - delta_temperature);
    printf("    heat source is group %d in grouping method 0.\n", source);
    printf("    heat sink is group %d in grouping method 0.\n", sink);
  } else {
    printf("Integrate with constant-power heating and cooling for this run.\n");
    printf("    choose the custom constant-power velocity-scaling method (heat_nhc_power).\n");
    printf("    average temperature is %g K (no thermostat acts on source/sink).\n", temperature);
    printf(
      "    tau_T is %g time_step (parsed but NOT used by this command).\n",
      temperature_coupling);
    printf(
      "    heating power is %g eV/fs (total for the whole group, not per atom).\n",
      delta_temperature);
    printf(
      "    every step, %g eV/fs * time_step is added to the source and removed from the "
      "sink.\n",
      delta_temperature);
    printf("    heat source is group %d in grouping method 0.\n", source);
    printf("    heat sink is group %d in grouping method 0.\n", sink);
  }
}

double Ensemble_NHC::get_temperature1() const
{
  return temperature1_;
}

double Ensemble_NHC::get_temperature2() const
{
  return temperature2_;
}

void Ensemble_NHC::initialize_run(
  const double time_step, Atom& atom, Box&, const std::vector<Group>& group)
{
  if (type == EnsembleType::NVT_NHC) {
    // position and momentum variables for one NHC
    pos_nhc1[0] = pos_nhc1[1] = pos_nhc1[2] = pos_nhc1[3] = 0.0;
    vel_nhc1[0] = vel_nhc1[2] = 1.0;
    vel_nhc1[1] = vel_nhc1[3] = -1.0;

    double tau = time_step * temperature_coupling;
    double kT = K_B * temperature;
    double dN = DIM * atom.number_of_atoms;
    for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; i++) {
      mas_nhc1[i] = kT * tau * tau;
    }
    mas_nhc1[0] *= dN;
    return;
  }

  // position and momentum variables for NHC
  pos_nhc1[0] = pos_nhc1[1] = pos_nhc1[2] = pos_nhc1[3] = 0.0;
  pos_nhc2[0] = pos_nhc2[1] = pos_nhc2[2] = pos_nhc2[3] = 0.0;
  vel_nhc1[0] = vel_nhc1[2] = vel_nhc2[0] = vel_nhc2[2] = 1.0;
  vel_nhc1[1] = vel_nhc1[3] = vel_nhc2[1] = vel_nhc2[3] = -1.0;

  double tau = time_step * temperature_coupling;
  double kT1 = K_B * (temperature + delta_temperature);
  double kT2 = K_B * (temperature - delta_temperature);
  double dN1 = DIM * group[0].cpu_size[source];
  double dN2 = DIM * group[0].cpu_size[sink];
  for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; i++) {
    mas_nhc1[i] = kT1 * tau * tau;
    mas_nhc2[i] = kT2 * tau * tau;
  }
  mas_nhc1[0] *= dN1;
  mas_nhc2[0] *= dN2;

  // initialize the energies transferred from the system to the baths
  energy_transferred[0] = 0.0;
  energy_transferred[1] = 0.0;

  initialize_group_kinetic_energy_workspace(group[0].number);
  initialize_group_com_velocity_workspace(group[0].number);
}

// The Nose-Hover thermostat integrator
// Run it on the CPU, which requires copying the kinetic energy
// from the GPU to the CPU
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
  // These constants are taken from Tuckerman's book
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

  double factor = 1.0; // to be accumulated

  for (int n1 = 0; n1 < n_sy; n1++) {
    double dt2 = dt2_particle * w[n1] / n_respa;
    double dt4 = dt2 * 0.5;
    double dt8 = dt4 * 0.5;
    for (int n2 = 0; n2 < n_respa; n2++) {

      // update velocity of the last (M - 1) thermostat:
      double G = vel_eta[M - 2] * vel_eta[M - 2] / mas_eta[M - 2] - kT;
      vel_eta[M - 1] += dt4 * G;

      // update thermostat velocities from M - 2 to 0:
      for (int m = M - 2; m >= 0; m--) {
        double tmp = exp(-dt8 * vel_eta[m + 1] / mas_eta[m + 1]);
        if (m == 0) {
          G = Ek2 - dN * kT;
        } else {
          G = vel_eta[m - 1] * vel_eta[m - 1] / mas_eta[m - 1] - kT;
        }
        vel_eta[m] = tmp * (tmp * vel_eta[m] + dt4 * G);
      }

      // update thermostat positions from M - 1 to 0:
      for (int m = M - 1; m >= 0; m--) {
        pos_eta[m] += dt2 * vel_eta[m] / mas_eta[m];
      }

      // compute the scale factor
      double factor_local = exp(-dt2 * vel_eta[0] / mas_eta[0]);
      Ek2 *= factor_local * factor_local;
      factor *= factor_local;

      // update thermostat velocities from 0 to M - 2:
      for (int m = 0; m < M - 1; m++) {
        double tmp = exp(-dt8 * vel_eta[m + 1] / mas_eta[m + 1]);
        if (m == 0) {
          G = Ek2 - dN * kT;
        } else {
          G = vel_eta[m - 1] * vel_eta[m - 1] / mas_eta[m - 1] - kT;
        }
        vel_eta[m] = tmp * (tmp * vel_eta[m] + dt4 * G);
      }

      // update velocity of the last (M - 1) thermostat:
      G = vel_eta[M - 2] * vel_eta[M - 2] / mas_eta[M - 2] - kT;
      vel_eta[M - 1] += dt4 * G;
    }
  }
  return factor;
}

void Ensemble_NHC::integrate_nvt_nhc_1(
  const double time_step,
  const double volume,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  const int number_of_atoms = mass.size();

  double kT = K_B * temperature;
  double dN = (double)DIM * number_of_atoms;
  double dt2 = time_step * 0.5;

  const int M = NOSE_HOOVER_CHAIN_LENGTH;
  find_thermo(volume, group, mass, potential_per_atom, velocity_per_atom, virial_per_atom, thermo);

  double ek2[1];
  thermo.copy_to_host(ek2, 1);
  ek2[0] *= DIM * number_of_atoms * K_B;
  double factor = nhc(M, pos_nhc1, vel_nhc1, mas_nhc1, ek2[0], kT, dN, dt2);
  scale_velocity_global(factor, velocity_per_atom);

  velocity_verlet(
    true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
}

void Ensemble_NHC::integrate_nvt_nhc_2(
  const double time_step,
  const double volume,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  const int number_of_atoms = mass.size();

  double kT = K_B * temperature;
  double dN = (double)DIM * number_of_atoms;
  double dt2 = time_step * 0.5;
  const int M = NOSE_HOOVER_CHAIN_LENGTH;
  double ek2[1];

  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  find_thermo(volume, group, mass, potential_per_atom, velocity_per_atom, virial_per_atom, thermo);

  thermo.copy_to_host(ek2, 1);
  ek2[0] *= DIM * number_of_atoms * K_B;
  double factor = nhc(M, pos_nhc1, vel_nhc1, mas_nhc1, ek2[0], kT, dN, dt2);
  scale_velocity_global(factor, velocity_per_atom);
}

// integrate by one step, with heating and cooling,
// using Nose-Hoover chain method
void Ensemble_NHC::integrate_heat_nhc_1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  int label_1 = source;
  int label_2 = sink;

  double kT1 = K_B * (temperature + delta_temperature);
  double kT2 = K_B * (temperature - delta_temperature);
  double dN1 = (double)DIM * group[0].cpu_size[source];
  double dN2 = (double)DIM * group[0].cpu_size[sink];
  double dt2 = time_step * 0.5;

  std::vector<double>& ek2 = group_kinetic_energy_cpu_;
  GPU_Vector<double>& vcx = group_com_velocity_x_;
  GPU_Vector<double>& vcy = group_com_velocity_y_;
  GPU_Vector<double>& vcz = group_com_velocity_z_;
  GPU_Vector<double>& ke = group_kinetic_energy_;

  // NHC first
  find_vc_and_ke(group, mass, velocity_per_atom, vcx.data(), vcy.data(), vcz.data(), ke.data());

  ke.copy_to_host(ek2.data());

  double factor_1 =
    nhc(NOSE_HOOVER_CHAIN_LENGTH, pos_nhc1, vel_nhc1, mas_nhc1, ek2[label_1], kT1, dN1, dt2);
  double factor_2 =
    nhc(NOSE_HOOVER_CHAIN_LENGTH, pos_nhc2, vel_nhc2, mas_nhc2, ek2[label_2], kT2, dN2, dt2);

  // accumulate the energies transferred from the system to the baths
  energy_transferred[0] += ek2[label_1] * 0.5 * (1.0 - factor_1 * factor_1);
  energy_transferred[1] += ek2[label_2] * 0.5 * (1.0 - factor_2 * factor_2);

  scale_velocity_local(
    factor_1, factor_2, vcx.data(), vcy.data(), vcz.data(), ke.data(), group, velocity_per_atom);

  velocity_verlet(
    true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
}

// integrate by one step, with heating and cooling,
// using Nose-Hoover chain method
void Ensemble_NHC::integrate_heat_nhc_2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  int label_1 = source;
  int label_2 = sink;

  double kT1 = K_B * (temperature + delta_temperature);
  double kT2 = K_B * (temperature - delta_temperature);
  double dN1 = (double)DIM * group[0].cpu_size[source];
  double dN2 = (double)DIM * group[0].cpu_size[sink];
  double dt2 = time_step * 0.5;

  std::vector<double>& ek2 = group_kinetic_energy_cpu_;
  GPU_Vector<double>& vcx = group_com_velocity_x_;
  GPU_Vector<double>& vcy = group_com_velocity_y_;
  GPU_Vector<double>& vcz = group_com_velocity_z_;
  GPU_Vector<double>& ke = group_kinetic_energy_;

  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  // NHC second
  find_vc_and_ke(group, mass, velocity_per_atom, vcx.data(), vcy.data(), vcz.data(), ke.data());

  ke.copy_to_host(ek2.data());
  double factor_1 =
    nhc(NOSE_HOOVER_CHAIN_LENGTH, pos_nhc1, vel_nhc1, mas_nhc1, ek2[label_1], kT1, dN1, dt2);
  double factor_2 =
    nhc(NOSE_HOOVER_CHAIN_LENGTH, pos_nhc2, vel_nhc2, mas_nhc2, ek2[label_2], kT2, dN2, dt2);

  // accumulate the energies transferred from the system to the baths
  energy_transferred[0] += ek2[label_1] * 0.5 * (1.0 - factor_1 * factor_1);
  energy_transferred[1] += ek2[label_2] * 0.5 * (1.0 - factor_2 * factor_2);

  scale_velocity_local(
    factor_1, factor_2, vcx.data(), vcy.data(), vcz.data(), ke.data(), group, velocity_per_atom);
}

// integrate by one step, with a constant heating/cooling power (custom command
// heat_nhc_power): every step, a fixed energy dE = power * time_step is added to
// the heat source and removed from the heat sink by scaling the velocities
// about the group center-of-mass velocity (hence no net momentum is injected).
// No Nose-Hoover chain thermostat acts on the source/sink here, so the
// accumulated energy_transferred is exactly the explicitly injected/removed
// energy; there is no additional thermostat work.
void Ensemble_NHC::integrate_heat_nhc_power_1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  velocity_verlet(
    true, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
}

void Ensemble_NHC::integrate_heat_nhc_power_2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  int label_1 = source;
  int label_2 = sink;

  std::vector<double>& ek2 = group_kinetic_energy_cpu_;
  GPU_Vector<double>& vcx = group_com_velocity_x_;
  GPU_Vector<double>& vcy = group_com_velocity_y_;
  GPU_Vector<double>& vcz = group_com_velocity_z_;
  GPU_Vector<double>& ke = group_kinetic_energy_;

  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  find_vc_and_ke(group, mass, velocity_per_atom, vcx.data(), vcy.data(), vcz.data(), ke.data());
  // delta_temperature stores the heating power P in eV/fs (total for the whole
  // group, not per atom) for this command; the energy exchanged per step is
  // P * dt_fs, where dt_fs = time_step (in GPUMD natural time units) converted
  // to fs via TIME_UNIT_CONVERSION
  double dE = delta_temperature * time_step * TIME_UNIT_CONVERSION;
  ke.copy_to_host(ek2.data());

  const double K_source = ek2[label_1] * 0.5; // COM-relative kinetic energy
  const double K_sink = ek2[label_2] * 0.5;
  // Guards for pathological states (e.g. starting from a uniform-temperature
  // configuration before a temperature gradient has been established): never
  // remove more kinetic energy than the sink group has, and never heat a
  // zero-kinetic-energy source group. The energy accounting below uses the
  // ACTUAL energies, so any activation of these guards shows up as a
  // deviation of the accumulated energies from +/- P * time.
  double dE_in = (K_source > 0.0) ? dE : 0.0;
  double dE_out = dE;
  if (dE_out > 0.99 * K_sink) {
    dE_out = 0.99 * K_sink;
  }
  if (dE_out < 0.0) {
    dE_out = 0.0;
  }
  double factor_1 = (K_source > 0.0) ? sqrt(1.0 + dE_in / K_source) : 1.0; // energy in
  double factor_2 = (K_sink > 0.0) ? sqrt(1.0 - dE_out / K_sink) : 0.0; // energy out

  // accumulate the energies transferred from the system to the baths
  energy_transferred[0] -= dE_in;
  energy_transferred[1] += dE_out;

  scale_velocity_local(
    factor_1, factor_2, vcx.data(), vcy.data(), vcz.data(), ke.data(), group, velocity_per_atom);
}

void Ensemble_NHC::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (type == EnsembleType::NVT_NHC) {
    integrate_nvt_nhc_1(
      time_step,
      box.get_volume(),
      group,
      atom.mass,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom,
      thermo);
  } else if (type == EnsembleType::HEAT_NHC_POWER) {
    integrate_heat_nhc_power_1(
      time_step,
      group,
      atom.mass,
      atom.force_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom);
  } else {
    integrate_heat_nhc_1(
      time_step,
      group,
      atom.mass,
      atom.force_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom);
  }
}

void Ensemble_NHC::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (type == EnsembleType::NVT_NHC) {
    integrate_nvt_nhc_2(
      time_step,
      box.get_volume(),
      group,
      atom.mass,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom,
      thermo);
  } else if (type == EnsembleType::HEAT_NHC_POWER) {
    integrate_heat_nhc_power_2(
      time_step,
      group,
      atom.mass,
      atom.force_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom);
  } else {
    integrate_heat_nhc_2(
      time_step,
      group,
      atom.mass,
      atom.force_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom);
  }
}
