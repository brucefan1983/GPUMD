/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
Spin ensemble implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com
*/

#include "ensemble.cuh"
#include "lattice.cuh"
#include "tspin.cuh"

Spin_Ensemble::~Spin_Ensemble() = default;
bool Spin_Ensemble::has_spin_mttk_driver() const
{
  return type == EnsembleType::MTTK && (use_spin_tspin);
}

bool Spin_Ensemble::has_spin_nhc_driver() const
{
  // A fixed lattice has no thermostat driver despite the NVT_NHC type tag.
  return type == EnsembleType::NVT_NHC && spin_lattice_enabled && (use_spin_tspin);
}

double Spin_Ensemble::spin_protocol_fraction(int step, int number_of_steps) const
{
  if (has_spin_mttk_driver())
    return static_cast<const Lattice_MTTK*>(lattice_.get())
      ->protocol_fraction(step, number_of_steps);
  if (has_spin_nhc_driver())
    return static_cast<const Lattice_NHC*>(lattice_.get())
      ->protocol_fraction(step, number_of_steps);
  return double(step) / number_of_steps;
}

void Spin_Ensemble::restore_time(const Atom& atom, double& global_time) const
{
  if (use_spin_tspin) {
    if (!atom.spin_lattice_restart.mttk.empty())
      global_time = atom.spin_lattice_restart.mttk_time;
    if (!atom.spin_lattice_restart.nhc.empty())
      global_time = atom.spin_lattice_restart.nhc_time;
  }
}

void Spin_Ensemble::initialize_run(
  double time_step, Atom& atom, Box& box, const std::vector<Group>& group)
{
  if (lattice_) {
    lattice_->fixed_group = fixed_group;
    lattice_->move_group = move_group;
    lattice_->fixed_grouping_method = fixed_grouping_method;
    lattice_->move_grouping_method = move_grouping_method;
    if (move_group >= 0)
      for (int i = 0; i < 3; ++i)
        lattice_->move_velocity[i] = move_velocity[i];
    lattice_->deform_x = deform_x;
    lattice_->deform_y = deform_y;
    lattice_->deform_z = deform_z;
    lattice_->deform_xy = deform_xy;
    lattice_->deform_xz = deform_xz;
    lattice_->deform_yz = deform_yz;
    lattice_->temperature = temperature;
  }
  if (!has_spin_mttk_driver() && !atom.spin_lattice_restart.mttk.empty()) {
    printf("Discarding MTTK restart state for a different ensemble.\n");
    atom.spin_lattice_restart.mttk.clear();
  }
  if (!has_spin_nhc_driver() && !atom.spin_lattice_restart.nhc.empty()) {
    printf("Discarding NVT NHC restart state for a different ensemble or lattice off.\n");
    atom.spin_lattice_restart.nhc.clear();
  }
  if (lattice_)
    lattice_->initialize_run(time_step, atom, box, group);
  spin_integrator.reset();
  if (use_spin_tspin) {
    spin_integrator.reset(new Spin_TSPIN(
      temperature1, temperature_coupling, time_step, spin_mass_factor_by_type, spin_seed, atom));
    spin_integrator->temperature = temperature;
  }
}

void Spin_Ensemble::initialize_before_first_step(
  double dt,
  int steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (lattice_)
    lattice_->initialize_before_first_step(dt, steps, group, box, atom, thermo);
}
void Spin_Ensemble::finalize_run(const Atom& atom, const Box& box)
{
  if (lattice_)
    lattice_->finalize_run(atom, box);
}
void Spin_Ensemble::set_step_temperature(int step, int steps)
{
  temperature = temperature1 + (temperature2 - temperature1) * spin_protocol_fraction(step, steps);
  if (lattice_)
    lattice_->temperature = temperature;
  spin_integrator->temperature = temperature;
}
void Spin_Ensemble::compute1(
  double dt,
  int step,
  int steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  set_step_temperature(step, steps);
  if (lattice_)
    lattice_->compute1(dt, step, steps, group, box, atom, thermo);
  spin_integrator->compute1(dt, atom);
}
void Spin_Ensemble::compute2(
  double dt,
  int step,
  int steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo,
  Force& force)
{
  set_step_temperature(step, steps);
  if (lattice_) {
    lattice_->compute2(dt, step, steps, group, box, atom, thermo, force);
    energy_transferred[0] = lattice_->energy_transferred[0];
    energy_transferred[1] = lattice_->energy_transferred[1];
    energy_transferred_n = lattice_->energy_transferred_n;
  } else {
    find_thermo(
      box.get_volume(),
      group,
      atom.mass,
      atom.potential_per_atom,
      atom.velocity_per_atom,
      atom.virial_per_atom,
      thermo);
  }
  spin_integrator->compute2(dt, atom);
}
