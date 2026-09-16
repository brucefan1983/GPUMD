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
The Bussi-Donadio-Parrinello thermostat:
[1] G. Bussi et al. J. Chem. Phys. 126, 014101 (2007).
------------------------------------------------------------------------------*/

#include "ensemble_bdp.cuh"
#include "svr_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <chrono>
#include <cstring>
#define DIM 3

void Ensemble_BDP::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

Ensemble_BDP::Ensemble_BDP(
  const char** param, int num_param, const std::vector<Group>& group)
{
  parse(param, num_param, group);
}

void Ensemble_BDP::parse_heat_groups(
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

void Ensemble_BDP::parse(
  const char** param, int num_param, const std::vector<Group>& group)
{
  if (strcmp(param[1], "nvt_bdp") == 0) {
    type = EnsembleType::NVT_BDP;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_bdp should have 3 parameters.");
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
    printf("    choose the Bussi-Donadio-Parrinello method.\n");
    printf("    initial temperature is %g K.\n", temperature1_);
    printf("    final temperature is %g K.\n", temperature2_);
    printf("    tau_T is %g time_step.\n", temperature_coupling);
    return;
  }

  if (strcmp(param[1], "heat_bdp") != 0) {
    PRINT_INPUT_ERROR("Invalid Bussi-Donadio-Parrinello ensemble type.");
  }
  type = EnsembleType::HEAT_BDP;
  if (num_param != 7) {
    PRINT_INPUT_ERROR("ensemble heat_bdp should have 5 parameters.");
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
  if (!is_valid_real(param[4], &delta_temperature)) {
    PRINT_INPUT_ERROR("Temperature difference should be a number.");
  }
  if (delta_temperature >= temperature || delta_temperature <= -temperature) {
    PRINT_INPUT_ERROR("|Temperature difference| is too large.");
  }

  parse_heat_groups(param, group);

  printf("Integrate with heating and cooling for this run.\n");
  printf("    choose the Bussi-Donadio-Parrinello method.\n");
  printf("    average temperature is %g K.\n", temperature);
  printf("    tau_T is %g time_step.\n", temperature_coupling);
  printf("    delta_T is %g K.\n", delta_temperature);
  printf("    T_hot is %g K.\n", temperature + delta_temperature);
  printf("    T_cold is %g K.\n", temperature - delta_temperature);
  printf("    heat source is group %d in grouping method 0.\n", source);
  printf("    heat sink is group %d in grouping method 0.\n", sink);
}

double Ensemble_BDP::get_temperature1() const
{
  return temperature1_;
}

double Ensemble_BDP::get_temperature2() const
{
  return temperature2_;
}

void Ensemble_BDP::initialize_run(
  const double, const Atom&, const std::vector<Group>& group)
{
  initialize_rng();
  if (type == EnsembleType::HEAT_BDP) {
    // initialize the energies transferred from the system to the baths
    energy_transferred[0] = 0.0;
    energy_transferred[1] = 0.0;
    initialize_group_kinetic_energy_workspace(group[0].number);
    initialize_group_com_velocity_workspace(group[0].number);
  }
}

Ensemble_BDP::~Ensemble_BDP(void)
{
  // nothing now
}

void Ensemble_BDP::integrate_nvt_bdp_2(
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

  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  // get thermo
  int N_fixed = (fixed_group == -1) ? 0 : group[fixed_grouping_method].cpu_size[fixed_group];
  N_fixed += (move_group == -1) ? 0 : group[move_grouping_method].cpu_size[move_group];
  find_thermo(volume, group, mass, potential_per_atom, velocity_per_atom, virial_per_atom, thermo);

  // re-scale the velocities
  double ek[1];
  thermo.copy_to_host(ek, 1);
  int ndeg = 3 * (number_of_atoms - N_fixed);
  ek[0] *= ndeg * K_B * 0.5; // from temperature to kinetic energy
  double sigma = ndeg * K_B * temperature * 0.5;
  double factor = resamplekin(ek[0], sigma, ndeg, temperature_coupling, rng);
  factor = sqrt(factor / ek[0]);
  scale_velocity_global(factor, velocity_per_atom);
}

// integrate by one step, with heating and cooling, using the BDP method
void Ensemble_BDP::integrate_heat_bdp_2(
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
  double dN1 = (double)DIM * (group[0].cpu_size[source] - 1);
  double dN2 = (double)DIM * (group[0].cpu_size[sink] - 1);
  double sigma_1 = dN1 * kT1 * 0.5;
  double sigma_2 = dN2 * kT2 * 0.5;

  std::vector<double>& ek = group_kinetic_energy_cpu_;
  GPU_Vector<double>& vcx = group_com_velocity_x_;
  GPU_Vector<double>& vcy = group_com_velocity_y_;
  GPU_Vector<double>& vcz = group_com_velocity_z_;
  GPU_Vector<double>& ke = group_kinetic_energy_;

  velocity_verlet(
    false, time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

  // get center of mass velocity and relative kinetic energy
  find_vc_and_ke(group, mass, velocity_per_atom, vcx.data(), vcy.data(), vcz.data(), ke.data());

  ke.copy_to_host(ek.data());
  ek[label_1] *= 0.5;
  ek[label_2] *= 0.5;

  // get the re-scaling factors
  double factor_1 = resamplekin(ek[label_1], sigma_1, dN1, temperature_coupling, rng);
  double factor_2 = resamplekin(ek[label_2], sigma_2, dN2, temperature_coupling, rng);
  factor_1 = sqrt(factor_1 / ek[label_1]);
  factor_2 = sqrt(factor_2 / ek[label_2]);

  // accumulate the energies transferred from the system to the baths
  energy_transferred[0] += ek[label_1] * (1.0 - factor_1 * factor_1);
  energy_transferred[1] += ek[label_2] * (1.0 - factor_2 * factor_2);

  scale_velocity_local(
    factor_1, factor_2, vcx.data(), vcy.data(), vcz.data(), ke.data(), group, velocity_per_atom);
}

void Ensemble_BDP::compute1(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  velocity_verlet(
    true,
    time_step,
    group,
    atom.mass,
    atom.force_per_atom,
    atom.position_per_atom,
    atom.velocity_per_atom);
}

void Ensemble_BDP::compute2(
  const double time_step,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (type == EnsembleType::NVT_BDP) {
    integrate_nvt_bdp_2(
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
  } else {
    integrate_heat_bdp_2(
      time_step,
      group,
      atom.mass,
      atom.force_per_atom,
      atom.position_per_atom,
      atom.velocity_per_atom);
  }
}
