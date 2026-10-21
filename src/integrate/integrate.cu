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

/*----------------------------------------------------------------------------80
The driver class for the various integrators.
------------------------------------------------------------------------------*/

#include "ensemble_bao.cuh"
#include "ensemble_bdp.cuh"
#include "ensemble_ber.cuh"
#include "ensemble_lan.cuh"
#include "ensemble_heat_hybrid.cuh"
#include "ensemble_msst.cuh"
#include "ensemble_mttk.cuh"
#include "ensemble_nhc.cuh"
#include "ensemble_nphug.cuh"
#include "ensemble_npt_qtb.cuh"
#include "ensemble_npt_scr.cuh"
#include "ensemble_nve.cuh"
#include "ensemble_pimd.cuh"
#include "ensemble_qtb.cuh"
#include "ensemble_ti.cuh"
#include "ensemble_ti_as.cuh"
#include "ensemble_ti_liquid.cuh"
#include "ensemble_ti_nep.cuh"
#include "ensemble_ti_rs.cuh"
#include "ensemble_ti_spring.cuh"
#include "ensemble_ttm.cuh"
#include "ensemble_wall_harmonic.cuh"
#include "ensemble_wall_mirror.cuh"
#include "ensemble_wall_piston.cuh"
#include "integrate.cuh"
#include "spin_sib.cuh"
#include "spin_glsd.cuh"
#include "spin_tspin.cuh"
#include "ensemble_fixed_lattice.cuh"
#include <algorithm>
#include <cmath>
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <utility>

namespace
{
bool is_tspin_option(const char* value)
{
  return strcmp(value, "lattice") == 0 ||
    strcmp(value, "mass_factor") == 0 || strcmp(value, "seed") == 0;
}

bool is_langevin_spin_option(const char* value)
{
  return strcmp(value, "lattice") == 0 || strcmp(value, "alpha") == 0 ||
    strcmp(value, "damping") == 0 || strcmp(value, "mobility") == 0 ||
    strcmp(value, "gamma") == 0 || strcmp(value, "stemp") == 0 ||
    strcmp(value, "seed") == 0;
}

void print_spin_mass_factors(
  const Atom& atom,
  const std::vector<double>& mass_factor_by_type)
{
  for (std::size_t type = 0; type < mass_factor_by_type.size(); ++type) {
    for (int atom_index = 0; atom_index < atom.number_of_atoms; ++atom_index) {
      if (atom.cpu_type[atom_index] == static_cast<int>(type)) {
        printf(
          "    spin mass factor for %s is %g.\n",
          atom.cpu_atom_symbol[atom_index].c_str(),
          mass_factor_by_type[type]);
        break;
      }
    }
  }
}
} // namespace

bool Integrate::has_ensemble() const
{
  return ensemble_ != nullptr;
}

EnsembleType Integrate::get_type() const
{
  return type;
}

int Integrate::get_fixed_group() const
{
  return fixed_group;
}

int Integrate::get_move_group() const
{
  return move_group;
}

int Integrate::get_fixed_grouping_method() const
{
  return fixed_grouping_method;
}

int Integrate::get_move_grouping_method() const
{
  return move_grouping_method;
}

double Integrate::get_temperature1() const
{
  return temperature1;
}

double Integrate::get_temperature2() const
{
  return temperature2;
}

double Integrate::get_temperature() const
{
  return temperature;
}

std::string Integrate::mttk_restart_state() const
{
  return type == EnsembleType::MTTK ? static_cast<const Ensemble_MTTK*>(ensemble_.get())->restart_state() : "";
}

bool Integrate::has_nhc_driver() const
{
  // lattice off uses Ensemble_Fixed_Lattice despite the NVT_NHC type tag.
  return type == EnsembleType::NVT_NHC &&
    (!(use_spin_tspin || use_spin_sib || use_spin_glsd) || spin_lattice_enabled);
}

std::string Integrate::nhc_restart_state() const
{
  return has_nhc_driver() ? static_cast<const Ensemble_NHC*>(ensemble_.get())->restart_state() : "";
}

double Integrate::get_target_temperature_of_step() const { return ensemble_->temperature; }

int Integrate::get_number_of_atoms_for_temperature(
  const int number_of_atoms, const std::vector<Group>& group) const
{
  return ensemble_->get_number_of_atoms_for_temperature(number_of_atoms, group);
}

int Integrate::get_num_target_pressure_components() const
{
  return num_target_pressure_components;
}

int Integrate::get_number_of_beads() const
{
  return number_of_beads;
}

const double* Integrate::get_energy_transferred() const
{
  return ensemble_->energy_transferred;
}

const std::vector<double>& Integrate::get_energy_transferred_n() const
{
  return ensemble_->energy_transferred_n;
}

void Integrate::find_thermo(
  const double volume,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& thermo)
{
  ensemble_->find_thermo(
    volume,
    group,
    mass,
    potential_per_atom,
    velocity_per_atom,
    virial_per_atom,
    thermo);
}

void Integrate::set_deform(
  int new_deform_x,
  int new_deform_y,
  int new_deform_z,
  int new_deform_xy,
  int new_deform_xz,
  int new_deform_yz)
{
  deform_x = new_deform_x;
  deform_y = new_deform_y;
  deform_z = new_deform_z;
  deform_xy = new_deform_xy;
  deform_xz = new_deform_xz;
  deform_yz = new_deform_yz;
}

void Integrate::initialize(
  double time_step,
  Atom& atom,
  Box& box,
  const std::vector<Group>& group)
{
  if (move_group >= 0) {
    if (fixed_group < 0) {
      PRINT_INPUT_ERROR("It is not allowed to have moving group but no fixed group.");
    }
    if (fixed_grouping_method != move_grouping_method) {
      PRINT_INPUT_ERROR("The fixed and moving groups must use the same grouping method.");
    }
    if (move_group == fixed_group) {
      PRINT_INPUT_ERROR("The fixed and moving groups cannot be the same.");
    }
    if (
      type != EnsembleType::NVT_BER && type != EnsembleType::NVT_NHC &&
      type != EnsembleType::NVT_BDP && type != EnsembleType::HEAT_LAN) {
      PRINT_INPUT_ERROR(
        "It is only allowed to use nvt_ber, nvt_nhc, or nvt_bdp with a moving group.");
    }
  }

  Ensemble& ensemble = *ensemble_;
  ensemble.fixed_group = fixed_group;
  ensemble.fixed_grouping_method = fixed_grouping_method;
  ensemble.move_grouping_method = move_grouping_method;
  ensemble.move_group = move_group;
  if (move_group >= 0) {
    for (int i = 0; i < 3; ++i) {
      ensemble.move_velocity[i] = move_velocity[i];
    }
  }
  ensemble.deform_x = deform_x;
  ensemble.deform_y = deform_y;
  ensemble.deform_z = deform_z;
  ensemble.deform_xy = deform_xy;
  ensemble.deform_xz = deform_xz;
  ensemble.deform_yz = deform_yz;
  if (type != EnsembleType::MTTK && !atom.mttk_restart.empty()) {
    printf("Discarding MTTK restart state for a different ensemble.\n");
    atom.mttk_restart.clear();
  }
  if (!has_nhc_driver() && !atom.nhc_restart.empty()) {
    printf("Discarding NVT NHC restart state for a different ensemble or lattice off.\n");
    atom.nhc_restart.clear();
  }
  ensemble.initialize_run(time_step, atom, box, group);
  spin_integrator.reset();
  if (!use_spin_glsd) atom.spin_glsd_step = atom.spin_glsd_seed = 0;
  if (use_spin_tspin) {
    spin_integrator.reset(new Spin_TSPIN(
      temperature1,
      temperature_coupling,
      time_step,
      spin_mass_factor_by_type,
      spin_seed,
      atom));
    spin_integrator->temperature = temperature;
  } else if (use_spin_sib) {
    spin_integrator.reset(new Spin_SIB(
      spin_sib_alpha,
      spin_dynamics_gamma,
      spin_dynamics_temperature,
      spin_dynamics_seed,
      atom));
    spin_integrator->temperature = temperature;
  } else if (use_spin_glsd) {
    spin_integrator.reset(new Spin_GLSD(
      spin_dynamics_gamma, spin_glsd_mobility, spin_dynamics_temperature,
      spin_dynamics_seed, atom));
    spin_integrator->temperature = temperature;
  }

}

void Integrate::finalize(const Atom& atom, const Box& box)
{
  if (has_ensemble()) {
    ensemble_->finalize_run(atom, box);
  }
  ensemble_.reset();
  spin_integrator.reset();
  use_spin_tspin = use_spin_sib = use_spin_glsd = false;
  type = EnsembleType::UNKNOWN;
  fixed_group = -1; // no group has an index of -1
  move_group = -1;
  fixed_grouping_method = 0;
  move_grouping_method = 0;
  deform_x = 0;
  deform_y = 0;
  deform_z = 0;
  deform_xy = 0;
  deform_xz = 0;
  deform_yz = 0;
}

void Integrate::compute1(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  Ensemble& ensemble = *ensemble_;
  if (step == 0 && type == EnsembleType::MTTK) {
    ensemble.initialize_before_first_step(
      time_step, number_of_steps, group, box, atom, thermo);
  }
  const double step_over_number_of_steps = type == EnsembleType::MTTK ?
    static_cast<Ensemble_MTTK&>(ensemble).protocol_fraction(step, number_of_steps) :
    (has_nhc_driver() ? static_cast<Ensemble_NHC&>(ensemble).protocol_fraction(step, number_of_steps) :
    double(step) / number_of_steps);
  if (
    type == EnsembleType::NVE || type == EnsembleType::RPMD ||
    type == EnsembleType::TRPMD) {
    ensemble.temperature = temperature2;
  } else if (is_standard_nvt(type) || is_standard_npt(type) || type == EnsembleType::PIMD ||
             (type == EnsembleType::MTTK && (use_spin_tspin || use_spin_sib || use_spin_glsd))) {
    ensemble.temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  if (step == 0 && type != EnsembleType::MTTK) {
    ensemble.initialize_before_first_step(
      time_step, number_of_steps, group, box, atom, thermo);
  }

  if (spin_integrator) spin_integrator->temperature = ensemble.temperature;
  temperature = ensemble.temperature;
  ensemble.compute1(time_step, step, number_of_steps, group, box, atom, thermo);
  if (spin_integrator) spin_integrator->compute1(time_step, atom);
}

void Integrate::compute2(
  const double time_step,
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo,
  Force& force)
{
  Ensemble& ensemble = *ensemble_;
  const double step_over_number_of_steps = type == EnsembleType::MTTK ?
    static_cast<Ensemble_MTTK&>(ensemble).protocol_fraction(step, number_of_steps) :
    (has_nhc_driver() ? static_cast<Ensemble_NHC&>(ensemble).protocol_fraction(step, number_of_steps) :
    double(step) / number_of_steps);
  if (
    type == EnsembleType::NVE || type == EnsembleType::RPMD ||
    type == EnsembleType::TRPMD) {
    ensemble.temperature = temperature2;
  } else if (is_standard_nvt(type) || is_standard_npt(type) || type == EnsembleType::PIMD ||
             (type == EnsembleType::MTTK && (use_spin_tspin || use_spin_sib || use_spin_glsd))) {
    ensemble.temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  if (spin_integrator) spin_integrator->temperature = ensemble.temperature;
  temperature = ensemble.temperature;
  ensemble.compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
  if (spin_integrator) spin_integrator->compute2(time_step, atom);
  // A barostat can change the box in compute2. Force::compute refreshes the float copy of the box
  // only at the next step, and the measurements at the end of this step read it.
  box.set_is_orthogonal();
}

void Integrate::parse_ensemble(
  const std::vector<std::string>& tokens,
  const Atom& atom,
  const Box& box,
  const std::vector<Group>& group)
{
  if (tokens.size() < 2) {
    PRINT_INPUT_ERROR("Keyword 'ensemble' requires an ensemble type.");
  }

  if (has_ensemble()) {
    PRINT_INPUT_ERROR("Only one ensemble can be specified before each run.");
  }

  use_spin_tspin = tokens[1] == "nvt_tspin" || tokens[1] == "npt_tspin";
  use_spin_sib = tokens[1] == "nve_sib" || tokens[1] == "nvt_sib" || tokens[1] == "npt_sib";
  use_spin_glsd = tokens[1] == "nve_glsd" || tokens[1] == "nvt_glsd" || tokens[1] == "npt_glsd";
  if (use_spin_tspin || use_spin_sib || use_spin_glsd) {
    parse_spin_ensemble(tokens, atom, box, group);
    return;
  }

  // 1. Determine the integration method
  if (tokens[1] == "nve") {
    ensemble_ = std::make_unique<Ensemble_NVE>(tokens);
    type = EnsembleType::NVE;
  } else if (tokens[1] == "nvt_ber" || tokens[1] == "npt_ber") {
    auto ensemble_ber = std::make_unique<Ensemble_BER>(tokens, box);
    type = ensemble_ber->type;
    temperature1 = ensemble_ber->get_temperature1();
    temperature2 = ensemble_ber->get_temperature2();
    temperature = temperature1;
    if (type == EnsembleType::NPT_BER) {
      num_target_pressure_components = ensemble_ber->get_num_target_pressure_components();
    }
    ensemble_ = std::move(ensemble_ber);
  } else if (
    tokens[1] == "nvt_nhc" || tokens[1] == "heat_nhc" ||
    tokens[1] == "heat_nhc_power") {
    auto ensemble_nhc = std::make_unique<Ensemble_NHC>(tokens, group);
    type = ensemble_nhc->type;
    temperature = ensemble_nhc->temperature;
    if (type == EnsembleType::NVT_NHC) {
      temperature1 = ensemble_nhc->get_temperature1();
      temperature2 = ensemble_nhc->get_temperature2();
    }
    ensemble_ = std::move(ensemble_nhc);
  } else if (tokens[1] == "nvt_lan" || tokens[1] == "heat_lan") {
    auto ensemble_lan = std::make_unique<Ensemble_LAN>(tokens, group);
    type = ensemble_lan->type;
    temperature = ensemble_lan->temperature;
    if (type == EnsembleType::NVT_LAN) {
      temperature1 = ensemble_lan->get_temperature1();
      temperature2 = ensemble_lan->get_temperature2();
    }
    ensemble_ = std::move(ensemble_lan);
  } else if (tokens[1] == "nvt_bdp" || tokens[1] == "heat_bdp") {
    auto ensemble_bdp = std::make_unique<Ensemble_BDP>(tokens, group);
    type = ensemble_bdp->type;
    temperature = ensemble_bdp->temperature;
    if (type == EnsembleType::NVT_BDP) {
      temperature1 = ensemble_bdp->get_temperature1();
      temperature2 = ensemble_bdp->get_temperature2();
    }
    ensemble_ = std::move(ensemble_bdp);
  } else if (tokens[1] == "nvt_bao") {
    auto ensemble_bao = std::make_unique<Ensemble_BAO>(tokens);
    type = ensemble_bao->type;
    temperature1 = ensemble_bao->get_temperature1();
    temperature2 = ensemble_bao->get_temperature2();
    temperature = temperature1;
    ensemble_ = std::move(ensemble_bao);
  } else if (tokens[1] == "nvt_qtb") {
    auto ensemble_qtb = std::make_unique<Ensemble_QTB>(tokens);
    type = ensemble_qtb->type;
    temperature1 = ensemble_qtb->get_temperature1();
    temperature2 = ensemble_qtb->get_temperature2();
    temperature = ensemble_qtb->temperature;
    ensemble_ = std::move(ensemble_qtb);
  } else if (tokens[1] == "npt_scr") {
    auto ensemble_scr = std::make_unique<Ensemble_NPT_SCR>(tokens, box);
    type = ensemble_scr->type;
    temperature1 = ensemble_scr->get_temperature1();
    temperature2 = ensemble_scr->get_temperature2();
    temperature = ensemble_scr->temperature;
    num_target_pressure_components = ensemble_scr->get_num_target_pressure_components();
    ensemble_ = std::move(ensemble_scr);
  } else if (
    tokens[1] == "nvt_mttk" || tokens[1] == "npt_mttk" ||
    tokens[1] == "nph_mttk") {
    type = EnsembleType::MTTK;
    auto ensemble_mttk = std::make_unique<Ensemble_MTTK>(tokens);
    temperature1 = ensemble_mttk->t_start;
    temperature2 = ensemble_mttk->t_stop;
    ensemble_ = std::move(ensemble_mttk);
  } else if (tokens[1] == "npt_qtb") {
    type = EnsembleType::NPT_QTB;
    auto ensemble_npt_qtb = std::make_unique<Ensemble_NPT_QTB>(tokens);
    temperature1 = ensemble_npt_qtb->t_start;
    temperature2 = ensemble_npt_qtb->t_stop;
    ensemble_ = std::move(ensemble_npt_qtb);
  } else if (tokens[1] == "heat_ttm") {
    auto ensemble_ttm =
      std::make_unique<Ensemble_TTM>(tokens, atom, box, group);
    type = ensemble_ttm->type;
    temperature = ensemble_ttm->temperature;
    ensemble_ = std::move(ensemble_ttm);
  } else if (tokens[1] == "ttm") {
    auto ensemble_ttm =
      std::make_unique<Ensemble_TTM>(tokens, atom, box, group);
    type = ensemble_ttm->type;
    temperature = ensemble_ttm->temperature;
    temperature1 = 0.0;
    temperature2 = 0.0;
    ensemble_ = std::move(ensemble_ttm);
  } else if (tokens[1] == "heat_hybrid") {
    auto ensemble_hybrid =
      std::make_unique<Ensemble_Heat_Hybrid>(tokens, group);
    type = ensemble_hybrid->type;
    temperature = ensemble_hybrid->temperature;
    ensemble_ = std::move(ensemble_hybrid);
  } else if (
    tokens[1] == "rpmd" || tokens[1] == "trpmd" || tokens[1] == "pimd" ||
    tokens[1] == "pimd_scr") {
    auto ensemble_pimd = std::make_unique<Ensemble_PIMD>(tokens, box);
    type = ensemble_pimd->type;
    number_of_beads = ensemble_pimd->get_number_of_beads();
    if (type == EnsembleType::PIMD) {
      temperature1 = ensemble_pimd->get_temperature1();
      temperature2 = ensemble_pimd->get_temperature2();
      temperature = ensemble_pimd->temperature;
      num_target_pressure_components =
        ensemble_pimd->get_num_target_pressure_components();
    }
    ensemble_ = std::move(ensemble_pimd);
  } else if (tokens[1] == "msst") {
    type = EnsembleType::MSST;
    ensemble_ = std::make_unique<Ensemble_MSST>(tokens);
  } else if (tokens[1] == "ti_spring") {
    type = EnsembleType::TI_SPRING;
    ensemble_ = std::make_unique<Ensemble_TI_Spring>(tokens);
  } else if (tokens[1] == "wall_piston") {
    type = EnsembleType::WALL_PISTON;
    ensemble_ = std::make_unique<Ensemble_wall_piston>(tokens);
  } else if (tokens[1] == "nphug") {
    type = EnsembleType::NPHUG;
    ensemble_ = std::make_unique<Ensemble_NPHug>(tokens);
  } else if (tokens[1] == "ti") {
    type = EnsembleType::TI;
    ensemble_ = std::make_unique<Ensemble_TI>(tokens);
  } else if (tokens[1] == "wall_mirror") {
    type = EnsembleType::WALL_MIRROR;
    ensemble_ = std::make_unique<Ensemble_wall_mirror>(tokens);
  } else if (tokens[1] == "ti_rs") {
    type = EnsembleType::TI_RS;
    ensemble_ = std::make_unique<Ensemble_TI_RS>(tokens);
  } else if (tokens[1] == "ti_as") {
    type = EnsembleType::TI_AS;
    ensemble_ = std::make_unique<Ensemble_TI_AS>(tokens);
  } else if (tokens[1] == "wall_harmonic") {
    type = EnsembleType::WALL_HARMONIC;
    ensemble_ = std::make_unique<Ensemble_wall_harmonic>(tokens);
  } else if (tokens[1] == "ti_liquid") {
    type = EnsembleType::TI_LIQUID;
    ensemble_ = std::make_unique<Ensemble_TI_Liquid>(tokens);
  } else if (tokens[1] == "ti_nep") {
    type = EnsembleType::TI_NEP;
    ensemble_ = std::make_unique<Ensemble_TI_Nep>(tokens);
  } else {
    PRINT_INPUT_ERROR("Invalid ensemble type.");
  }
}

void Integrate::parse_fix(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("Keyword 'fix' should have 1 or 2 parameters.");
  }

  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot use 'fix' without grouping method.");
  }

  if (num_param == 3) {
    // fix grouping_method group_id
    if (!is_valid_int(tokens[1], &fixed_grouping_method)) {
      PRINT_INPUT_ERROR("Grouping method for 'fix' should be an integer.");
    }
    if (fixed_grouping_method < 0) {
      PRINT_INPUT_ERROR("Grouping method for 'fix' should >= 0.");
    }
    if (fixed_grouping_method >= group.size()) {
      PRINT_INPUT_ERROR("Grouping method for 'fix' should < number of grouping methods.");
    }
    if (!is_valid_int(tokens[2], &fixed_group)) {
      PRINT_INPUT_ERROR("Fixed group ID should be an integer.");
    }
  } else {
    // fix group_id (default grouping_method = 0)
    fixed_grouping_method = 0;
    if (!is_valid_int(tokens[1], &fixed_group)) {
      PRINT_INPUT_ERROR("Fixed group ID should be an integer.");
    }
  }

  if (fixed_group < 0) {
    PRINT_INPUT_ERROR("Fixed group ID should >= 0.");
  }

  if (fixed_group >= group[fixed_grouping_method].number) {
    PRINT_INPUT_ERROR("Fixed group ID should < number of groups.");
  }

  printf("Group %d in grouping method %d will be fixed.\n", fixed_group, fixed_grouping_method);
}

void Integrate::parse_move(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  if (num_param != 5 && num_param != 6) {
    PRINT_INPUT_ERROR("Keyword 'move' should have 4 or 5 parameters.");
  }

  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot use 'move' without grouping method.");
  }

  int vid; // index where vx starts
  if (num_param == 6) {
    // move grouping_method group_id vx vy vz
    if (!is_valid_int(tokens[1], &move_grouping_method)) {
      PRINT_INPUT_ERROR("Grouping method for 'move' should be an integer.");
    }
    if (move_grouping_method < 0) {
      PRINT_INPUT_ERROR("Grouping method for 'move' should >= 0.");
    }
    if (move_grouping_method >= group.size()) {
      PRINT_INPUT_ERROR("Grouping method for 'move' should < number of grouping methods.");
    }
    if (!is_valid_int(tokens[2], &move_group)) {
      PRINT_INPUT_ERROR("Moving group ID should be an integer.");
    }
    vid = 3;
  } else {
    // move group_id vx vy vz (default grouping_method = 0)
    move_grouping_method = 0;
    if (!is_valid_int(tokens[1], &move_group)) {
      PRINT_INPUT_ERROR("Moving group ID should be an integer.");
    }
    vid = 2;
  }

  if (move_group < 0) {
    PRINT_INPUT_ERROR("Moving group ID should >= 0.");
  }

  if (move_group >= group[move_grouping_method].number) {
    PRINT_INPUT_ERROR("Moving group ID should < number of groups.");
  }

  if (!is_valid_real(tokens[vid], &move_velocity[0])) {
    PRINT_INPUT_ERROR("Moving velocity in x direction should be a number.");
  }
  if (!is_valid_real(tokens[vid + 1], &move_velocity[1])) {
    PRINT_INPUT_ERROR("Moving velocity in y direction should be a number.");
  }
  if (!is_valid_real(tokens[vid + 2], &move_velocity[2])) {
    PRINT_INPUT_ERROR("Moving velocity in z direction should be a number.");
  }

  printf(
    "Group %d in grouping method %d will move with velocity vector (%g, %g, %g) A/fs.\n",
    move_group,
    move_grouping_method,
    move_velocity[0],
    move_velocity[1],
    move_velocity[2]);

  for (int d = 0; d < 3; ++d) {
    move_velocity[d] *= TIME_UNIT_CONVERSION; // natural to A/fs
  }
}

void Integrate::parse_spin_ensemble(
  const std::vector<std::string>& tokens, const Atom& atom,
  const Box& box, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  std::vector<const char*> args;
  for (const auto& token : tokens) args.push_back(token.c_str());
  const char** param = args.data();
  const bool spin_only = tokens[1] == "nve_sib" || tokens[1] == "nve_glsd";
  spin_lattice_enabled = !spin_only;
  spin_mass_factor_by_type.assign(atom.cpu_type_size.size(), 1.0);
  spin_seed = 12345;
  spin_sib_alpha = 0.0;
  spin_dynamics_gamma = 2.0 * 1000.0 / (HBAR * TIME_UNIT_CONVERSION);
  spin_dynamics_temperature = spin_only ? -1.0 : 0.0;
  spin_dynamics_seed = 12345;
  spin_glsd_damping = spin_glsd_mobility = 0.0;
  const bool npt = tokens[1] == "npt_tspin" || tokens[1] == "npt_sib" || tokens[1] == "npt_glsd";
  int spin_option_start = spin_only ? 2 : 5;
  if (npt) {
    spin_option_start = num_param;
    for (int i = 2; i < num_param; ++i) {
      if (use_spin_tspin ? is_tspin_option(param[i]) : is_langevin_spin_option(param[i])) {
        spin_option_start = i;
        break;
      }
    }
  }
  if (num_param < spin_option_start) PRINT_INPUT_ERROR("Missing spin ensemble parameters.");
  if (use_spin_tspin) {
    bool has_lattice = false;
    bool has_mass_factor = false;
    bool has_seed = false;
    int index = spin_option_start;
    while (index < num_param) {
      if (strcmp(param[index], "lattice") == 0) {
        if (index + 1 >= num_param) {
          PRINT_INPUT_ERROR("TSPIN lattice requires on or off.");
        }
        if (strcmp(param[1], "nvt_tspin") != 0) {
          PRINT_INPUT_ERROR("TSPIN lattice is only valid for nvt_tspin.");
        }
        if (has_lattice) {
          PRINT_INPUT_ERROR("TSPIN lattice cannot be repeated.");
        }
        has_lattice = true;
        if (strcmp(param[index + 1], "on") == 0) {
          spin_lattice_enabled = true;
        } else if (strcmp(param[index + 1], "off") == 0) {
          spin_lattice_enabled = false;
        } else {
          PRINT_INPUT_ERROR("TSPIN lattice should be on or off.");
        }
        index += 2;
      } else if (strcmp(param[index], "mass_factor") == 0) {
        if (has_mass_factor) {
          PRINT_INPUT_ERROR("TSPIN mass_factor cannot be repeated.");
        }
        has_mass_factor = true;
        if (index + 1 >= num_param) {
          PRINT_INPUT_ERROR("TSPIN mass_factor requires a value or element-value pairs.");
        }
        double common_factor = 0.0;
        if (is_valid_real(param[index + 1], &common_factor)) {
          if (!std::isfinite(common_factor) || common_factor <= 0.0) {
            PRINT_INPUT_ERROR("Scalar TSPIN mass_factor should be finite and > 0.");
          }
          std::fill(
            spin_mass_factor_by_type.begin(),
            spin_mass_factor_by_type.end(),
            common_factor);
          index += 2;
        } else {
          std::vector<int> configured(atom.cpu_type_size.size(), 0);
          ++index;
          while (index < num_param && !is_tspin_option(param[index])) {
            if (index + 1 >= num_param || is_tspin_option(param[index + 1])) {
              PRINT_INPUT_ERROR(
                "Element-wise TSPIN mass_factor requires element-value pairs.");
            }
            int type = -1;
            for (int atom_index = 0; atom_index < atom.number_of_atoms; ++atom_index) {
              if (atom.cpu_atom_symbol[atom_index] == param[index]) {
                type = atom.cpu_type[atom_index];
                break;
              }
            }
            if (type < 0) {
              PRINT_INPUT_ERROR("Unknown element in TSPIN mass_factor.");
            }
            if (configured[type] != 0) {
              PRINT_INPUT_ERROR("Element in TSPIN mass_factor cannot be repeated.");
            }
            double factor = 0.0;
            if (!is_valid_real(param[index + 1], &factor) ||
                !std::isfinite(factor) || factor < 0.0) {
              PRINT_INPUT_ERROR(
                "Element-wise TSPIN mass_factor should be finite and >= 0.");
            }
            spin_mass_factor_by_type[type] = factor;
            configured[type] = 1;
            index += 2;
          }
          for (int atom_index = 0; atom_index < atom.number_of_atoms; ++atom_index) {
            if (configured[atom.cpu_type[atom_index]] == 0) {
              PRINT_INPUT_ERROR(
                "Element-wise TSPIN mass_factor must list every element in model.xyz.");
            }
          }
        }
      } else if (strcmp(param[index], "seed") == 0) {
        if (index + 1 >= num_param) {
          PRINT_INPUT_ERROR("TSPIN seed requires a value.");
        }
        if (has_seed) {
          PRINT_INPUT_ERROR("TSPIN seed cannot be repeated.");
        }
        has_seed = true;
        if (!is_valid_int(param[index + 1], &spin_seed) || spin_seed <= 0) {
          PRINT_INPUT_ERROR("TSPIN seed should be a positive integer.");
        }
        index += 2;
      } else {
        PRINT_INPUT_ERROR("Unknown TSPIN optional keyword.");
      }
    }
  }

  if (use_spin_sib || use_spin_glsd) {
    bool has_lattice = false;
    bool has_damping = false;
    bool has_alpha = false;
    bool has_gamma = false;
    bool has_spin_temperature = false;
    bool has_seed = false;
    int index = spin_option_start;
    while (index < num_param) {
      if (index + 1 >= num_param) {
        PRINT_INPUT_ERROR("SIB/GLSD optional keywords require a value.");
      }
      if (strcmp(param[index], "lattice") == 0) {
        if (strcmp(param[1], "nvt_sib") != 0 && strcmp(param[1], "nvt_glsd") != 0) {
          PRINT_INPUT_ERROR("lattice is only valid for nvt_sib or nvt_glsd.");
        }
        if (has_lattice) {
          PRINT_INPUT_ERROR("SIB/GLSD lattice cannot be repeated.");
        }
        has_lattice = true;
        if (strcmp(param[index + 1], "on") == 0) {
          spin_lattice_enabled = true;
        } else if (strcmp(param[index + 1], "off") == 0) {
          spin_lattice_enabled = false;
        } else {
          PRINT_INPUT_ERROR("SIB/GLSD lattice should be on or off.");
        }
      } else if (strcmp(param[index], "damping") == 0) {
        if (!use_spin_glsd || has_damping || !is_valid_real(param[index + 1], &spin_glsd_damping) ||
            !std::isfinite(spin_glsd_damping) || spin_glsd_damping < 0.0) {
          PRINT_INPUT_ERROR("GLSD requires finite dimensionless damping >= 0 exactly once; SIB does not accept damping.");
        }
        has_damping = true;
      } else if (strcmp(param[index], "mobility") == 0) {
        PRINT_INPUT_ERROR("mobility is not supported; GLSD uses dimensionless damping = hbar * gamma_s_prime.");
      } else if (strcmp(param[index], "alpha") == 0) {
        if (use_spin_glsd || has_alpha) {
          PRINT_INPUT_ERROR("alpha belongs only to SIB and cannot be repeated.");
        }
        has_alpha = true;
        if (!is_valid_real(param[index + 1], &spin_sib_alpha) ||
            !std::isfinite(spin_sib_alpha) || spin_sib_alpha < 0.0) {
          PRINT_INPUT_ERROR("SIB alpha should be finite and >= 0.");
        }
      } else if (strcmp(param[index], "gamma") == 0) {
        if (has_gamma) {
          PRINT_INPUT_ERROR("SIB/GLSD gamma cannot be repeated.");
        }
        has_gamma = true;
        if (!is_valid_real(param[index + 1], &spin_dynamics_gamma) ||
            !std::isfinite(spin_dynamics_gamma) || spin_dynamics_gamma <= 0.0) {
          PRINT_INPUT_ERROR("SIB/GLSD gamma should be finite and > 0.");
        }
      } else if (strcmp(param[index], "stemp") == 0) {
        if (has_spin_temperature) {
          PRINT_INPUT_ERROR("SIB/GLSD stemp cannot be repeated.");
        }
        has_spin_temperature = true;
        if (!is_valid_real(param[index + 1], &spin_dynamics_temperature) ||
            !std::isfinite(spin_dynamics_temperature) ||
            (spin_dynamics_temperature < 0.0 && spin_dynamics_temperature != -1.0)) {
          PRINT_INPUT_ERROR("SIB/GLSD stemp should be -1, 0, or a positive temperature.");
        }
      } else if (strcmp(param[index], "seed") == 0) {
        if (has_seed) {
          PRINT_INPUT_ERROR("SIB/GLSD seed cannot be repeated.");
        }
        has_seed = true;
        if (!is_valid_int(param[index + 1], &spin_dynamics_seed) || spin_dynamics_seed <= 0) {
          PRINT_INPUT_ERROR("SIB/GLSD seed should be a positive integer.");
        }
      } else {
        PRINT_INPUT_ERROR("Unknown SIB/GLSD optional keyword.");
      }
      index += 2;
    }
    if (use_spin_glsd) {
      if (!has_damping) PRINT_INPUT_ERROR("GLSD requires explicit dimensionless damping.");
      // gamma = g_L / hbar, eta = hbar * gamma_s_prime, hence lambda = eta * gamma^2 * hbar.
      // Convert after parsing all options so keyword order cannot affect the model.
      const double hbar_ps = HBAR * TIME_UNIT_CONVERSION / 1000.0;
      spin_glsd_mobility = spin_glsd_damping * spin_dynamics_gamma * (spin_dynamics_gamma * hbar_ps);
      if (!std::isfinite(spin_glsd_mobility) ||
          (spin_glsd_damping > 0.0 && spin_glsd_mobility == 0.0)) {
        PRINT_INPUT_ERROR("GLSD damping/gamma conversion exceeds the representable mobility range.");
      }
    }
    if (spin_only && spin_dynamics_temperature == 0.0) {
      PRINT_INPUT_ERROR(
        "SIB/GLSD stemp 0 requires a temperature-controlled ensemble; use stemp -1 "
        "or a positive temperature with nve_sib/nve_glsd.");
    }
  }


  std::vector<std::string> lattice(tokens.begin(), tokens.begin() + spin_option_start);
  if (npt) {
    lattice[1] = "npt_mttk";
    auto mttk = std::make_unique<Ensemble_MTTK>(lattice);
    temperature1 = mttk->t_start;
    temperature2 = mttk->t_stop;
    temperature = temperature1;
    temperature_coupling = mttk->get_temperature_period();
    type = EnsembleType::MTTK;
    ensemble_ = std::move(mttk);
  } else if (spin_only) {
    temperature = temperature1 = temperature2 = 0.0;
    type = EnsembleType::NVE;
    ensemble_ = std::make_unique<Ensemble_Fixed_Lattice>(type);
  } else {
    lattice[1] = "nvt_nhc";
    auto nhc = std::make_unique<Ensemble_NHC>(lattice, group);
    temperature1 = nhc->get_temperature1();
    temperature2 = nhc->get_temperature2();
    temperature = temperature1;
    temperature_coupling = get_double_from_token(tokens[4], __FILE__, __LINE__);
    type = EnsembleType::NVT_NHC;
    if (spin_lattice_enabled) ensemble_ = std::move(nhc);
    else ensemble_ = std::make_unique<Ensemble_Fixed_Lattice>(type);
  }
  if (use_spin_tspin) {
    printf("Integrate spins with TSPIN.\n");
    print_spin_mass_factors(atom, spin_mass_factor_by_type);
  } else {
    if (use_spin_glsd) {
      printf("Integrate variable-magnitude spins with Ma-Dudarev GLSD (Heun).\n");
      printf("    damping = %.17g (dimensionless hbar * gamma_s_prime).\n", spin_glsd_damping);
      printf("    converted mobility = %.17g muB^2/(eV ps); Cartesian Boltzmann measure.\n", spin_glsd_mobility);
    } else {
      printf("Integrate spins with the semi-implicit B (SIB) method.\n");
      printf("    alpha = %g.\n", spin_sib_alpha);
    }
    printf("    gamma = %g (eV ps)^-1, stemp = %g K, seed = %d.\n",
      spin_dynamics_gamma, spin_dynamics_temperature, spin_dynamics_seed);
  }
}
