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
#include "ensemble_ti_rs.cuh"
#include "ensemble_ti_spring.cuh"
#include "ensemble_ttm.cuh"
#include "ensemble_wall_harmonic.cuh"
#include "ensemble_wall_mirror.cuh"
#include "ensemble_wall_piston.cuh"
#include "integrate.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <utility>

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
  ensemble.initialize_run(time_step, atom, box, group);
}

void Integrate::finalize(const Atom& atom, const Box& box)
{
  if (has_ensemble()) {
    ensemble_->finalize_run(atom, box);
  }
  ensemble_.reset();
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
  const double step_over_number_of_steps = double(step) / number_of_steps;
  if (
    type == EnsembleType::NVE || type == EnsembleType::RPMD ||
    type == EnsembleType::TRPMD) {
    ensemble.temperature = temperature2;
  } else if (is_standard_nvt(type) || is_standard_npt(type) || type == EnsembleType::PIMD) {
    ensemble.temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  if (step == 0) {
    ensemble.initialize_before_first_step(
      time_step, number_of_steps, group, box, atom, thermo);
  }

  ensemble.compute1(time_step, step, number_of_steps, group, box, atom, thermo);
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
  const double step_over_number_of_steps = double(step) / number_of_steps;
  if (
    type == EnsembleType::NVE || type == EnsembleType::RPMD ||
    type == EnsembleType::TRPMD) {
    ensemble.temperature = temperature2;
  } else if (is_standard_nvt(type) || is_standard_npt(type) || type == EnsembleType::PIMD) {
    ensemble.temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  ensemble.compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
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
