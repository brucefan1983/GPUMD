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

#include "glsd.cuh"
#include "integrator.cuh"
#include "sib.cuh"
#include "spin_ensemble.cuh"
#include "tspin.cuh"

Spin_Ensemble::~Spin_Ensemble() = default;
void Spin_Ensemble::initialize_run(
  double time_step, Atom& atom, Box& box, const std::vector<Group>& group)
{
  if (force_.get_number_of_potentials() != 1)
    PRINT_INPUT_ERROR("A spin ensemble requires one NEP-spin potential.");
  potential_ = dynamic_cast<Spin_Potential*>(&force_.get_potential(0));
  if (!potential_)
    PRINT_INPUT_ERROR("A spin ensemble requires a NEP-spin potential.");
  const auto& spin_types = potential_->spin_types();

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
  if (lattice_)
    lattice_->initialize_run(time_step, atom, box, group);
  spin_integrator.reset();
  if (use_spin_tspin) {
    spin_integrator.reset(new Spin_TSPIN(
      temperature1,
      temperature_coupling,
      time_step,
      spin_mass_factor_by_type,
      spin_seed,
      spin_types,
      atom));
    spin_integrator->temperature = temperature;
  } else if (use_spin_sib) {
    spin_integrator.reset(new Spin_SIB(
      spin_sib_alpha, spin_dynamics_gamma, spin_dynamics_temperature, spin_dynamics_seed, atom));
    spin_integrator->temperature = temperature;
  } else if (use_spin_glsd) {
    spin_integrator.reset(new Spin_GLSD(
      spin_dynamics_gamma,
      spin_glsd_mobility,
      spin_dynamics_temperature,
      spin_glsd_midpoint_tol,
      spin_dynamics_seed,
      spin_types,
      atom));
    spin_integrator->temperature = temperature;
  }

  if (!use_spin_tspin) {
    const size_t n = atom.number_of_atoms;
    modifiers_.force.resize(3 * n);
    modifiers_.mforce.resize(3 * n);
    modifiers_.energy.resize(n);
    modifiers_.virial.resize(9 * n);
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
  if (use_spin_glsd)
    static_cast<Spin_GLSD&>(*spin_integrator).report_iterations();
  if (lattice_)
    lattice_->finalize_run(atom, box);
}
void Spin_Ensemble::set_step_temperature(int step, int steps)
{
  temperature = temperature1 + (temperature2 - temperature1) * (double(step) / steps);
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
  // Capture the next potential evaluation before Actions add their increments.
  // The request is consumed there, so the corrector cannot overwrite it.
  if (!use_spin_tspin)
    potential_->request_force_snapshot(modifiers_);
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
  if (use_spin_tspin)
    spin_integrator->compute2(dt, atom);
  if (!use_spin_tspin) {
    box.set_is_orthogonal();
    atom.update_unwrapped_position(box);
    refresh_endpoint(dt, force, box, atom, group, thermo);
  }
}
