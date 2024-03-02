/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
#include <chrono>
#define DIM 3

void Ensemble_BDP::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

Ensemble_BDP::Ensemble_BDP(int t, int mg, double* mv, double T, double Tc)
{
  type = t;
  move_group = mg;
  move_velocity[0] = mv[0];
  move_velocity[1] = mv[1];
  move_velocity[2] = mv[2];
  temperature = T;
  temperature_coupling = Tc;
  initialize_rng();
}

Ensemble_BDP::Ensemble_BDP(int t, int source_input, int sink_input, double T, double Tc, double dT)
{
  type = t;
  temperature = T;
  temperature_coupling = Tc;
  delta_temperature = dT;
  source = source_input;
  sink = sink_input;
  // initialize the energies transferred from the system to the baths
  energy_transferred[0] = 0.0;
  energy_transferred[1] = 0.0;
  initialize_rng();
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
  int N_fixed = (fixed_group == -1) ? 0 : group[0].cpu_size[fixed_group];
  N_fixed += (move_group == -1) ? 0 : group[0].cpu_size[move_group];
  find_thermo(
    true, volume, group, mass, potential_per_atom, velocity_per_atom, virial_per_atom, thermo);

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
  int Ng = group[0].number;

  double kT1 = K_B * (temperature + delta_temperature);
  double kT2 = K_B * (temperature - delta_temperature);
  double dN1 = (double)DIM * (group[0].cpu_size[source] - 1);
  double dN2 = (double)DIM * (group[0].cpu_size[sink] - 1);
  double sigma_1 = dN1 * kT1 * 0.5;
  double sigma_2 = dN2 * kT2 * 0.5;

  // allocate some memory
  std::vector<double> ek(Ng);
  GPU_Vector<double> vcx(Ng), vcy(Ng), vcz(Ng), ke(Ng);

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
  if (type == 4) {
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
