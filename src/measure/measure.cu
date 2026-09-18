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
The driver class dealing with measurement.
------------------------------------------------------------------------------*/

#include "measure.cuh"
#include "active.cuh"
#include "add_efield.cuh"
#include "add_force.cuh"
#include "add_random_force.cuh"
#include "add_spring.cuh"
#include "adf.cuh"
#include "angular_rdf.cuh"
#include "compute.cuh"
#include "compute_chunk.cuh"
#include "compute_dpdt.cuh"
#include "compute_es.cuh"
#include "deform.cuh"
#include "dos.cuh"
#include "dump_beads.cuh"
#include "dump_cg.cuh"
#include "dump_dipole.cuh"
#include "dump_netcdf.cuh"
#include "dump_observer.cuh"
#include "dump_polarizability.cuh"
#include "dump_restart.cuh"
#include "dump_shock_nemd.cuh"
#include "dump_thermo.cuh"
#include "dump_xyz.cuh"
#include "electron_stop.cuh"
#include "extrapolation.cuh"
#include "hac.cuh"
#include "hnemdec_kappa.cuh"
#include "hnemd_kappa.cuh"
#include "iron_conductivity.cuh"
#include "lsqt.cuh"
#include "modal_analysis.cuh"
#include "msd.cuh"
#include "orientorder.cuh"
#include "plumed.cuh"
#include "rdf.cuh"
#include "sdc.cuh"
#include "shc.cuh"
#include "viscosity.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "mc/mc.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include <iostream>
#include <string>
#include <utility>
#include <vector>

bool Measure::parse_action(
  const std::vector<std::string>& tokens,
  const int number_of_types,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (tokens[0] == "dump_thermo") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Thermo(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_position") {
    PRINT_INPUT_ERROR(
      "dump_position has been removed. "
      "Use dump_xyz <interval> <filename> instead.");
  } else if (tokens[0] == "dump_netcdf") {
#ifdef USE_NETCDF
    std::vector<const char*> param;
    param.reserve(tokens.size());
    for (const auto& token : tokens) {
      param.emplace_back(token.c_str());
    }
    const int num_param = tokens.size();
    std::unique_ptr<Action> action;
    action.reset(new DUMP_NETCDF(param.data(), num_param, group, atom));
    actions_.emplace_back(std::move(action));
#else
    PRINT_INPUT_ERROR("dump_netcdf is available only when USE_NETCDF flag is set.\n");
#endif
  } else if (tokens[0] == "plumed") {
#ifdef USE_PLUMED
    std::vector<const char*> param;
    param.reserve(tokens.size());
    for (const auto& token : tokens) {
      param.emplace_back(token.c_str());
    }
    const int num_param = tokens.size();
    std::unique_ptr<Action> action;
    action.reset(new PLUMED(param.data(), num_param));
    actions_.emplace_back(std::move(action));
#else
    PRINT_INPUT_ERROR("plumed is available only when USE_PLUMED flag is set.\n");
#endif
  } else if (tokens[0] == "dump_restart") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Restart(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_velocity") {
    PRINT_INPUT_ERROR(
      "dump_velocity has been removed. "
      "Use dump_xyz <interval> <filename> velocity instead.");
  } else if (tokens[0] == "dump_force") {
    PRINT_INPUT_ERROR(
      "dump_force has been removed. "
      "Use dump_xyz <interval> <filename> force instead.");
  } else if (tokens[0] == "dump_exyz") {
    PRINT_INPUT_ERROR(
      "dump_exyz has been removed. "
      "Use dump_xyz <interval> <filename> velocity force potential instead.");
  } else if (tokens[0] == "dump_xyz") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_XYZ(tokens, group, atom));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_cg") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_CG(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_beads") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Beads(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_observer") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Observer(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_shock_nemd") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Shock_NEMD(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_dipole") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Dipole(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "dump_polarizability") {
    std::unique_ptr<Action> action;
    action.reset(new Dump_Polarizability(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "active") {
    std::unique_ptr<Action> action;
    action.reset(new Active(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_extrapolation") {
    std::unique_ptr<Action> action;
    action.reset(new Extrapolation(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_dos") {
    std::unique_ptr<Action> action;
    action.reset(new DOS(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_sdc") {
    std::unique_ptr<Action> action;
    action.reset(new SDC(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_msd") {
    std::unique_ptr<Action> action;
    action.reset(new MSD(tokens, group, atom));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_ic") {
    std::unique_ptr<Action> action;
    action.reset(new IC(tokens, atom));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_rdf") {
    std::unique_ptr<Action> action;
    action.reset(new RDF(tokens, box, atom.cpu_type_size));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_adf") {
    std::unique_ptr<Action> action;
    action.reset(new ADF(tokens, box, number_of_types));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_orientorder") {
    std::unique_ptr<Action> action;
    action.reset(new OrientOrder(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_angular_rdf") {
    std::unique_ptr<Action> action;
    action.reset(new AngularRDF(tokens, box, number_of_types));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_dpdt") {
    std::unique_ptr<Action> action;
    action.reset(new Compute_dpdt(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_es") {
    std::unique_ptr<Action> action;
    action.reset(new Compute_es(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_hac") {
    std::unique_ptr<Action> action;
    action.reset(new HAC(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_viscosity") {
    std::unique_ptr<Action> action;
    action.reset(new Viscosity(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_hnemd") {
    std::unique_ptr<Action> action;
    action.reset(new HNEMD(tokens, force));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_hnemdec") {
    std::unique_ptr<Action> action;
    action.reset(new HNEMDEC(tokens));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_shc") {
    std::unique_ptr<Action> action;
    action.reset(new SHC(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_gkma") {
    std::unique_ptr<Action> action;
    action.reset(new MODAL_ANALYSIS(tokens, number_of_types, 0, force));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_hnema") {
    std::unique_ptr<Action> action;
    action.reset(new MODAL_ANALYSIS(tokens, number_of_types, 1, force));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "deform") {
    Deform* deform = new Deform(tokens);
    integrate.set_deform(
      deform->get_deform_x(),
      deform->get_deform_y(),
      deform->get_deform_z(),
      deform->get_deform_xy(),
      deform->get_deform_xz(),
      deform->get_deform_yz());
    std::unique_ptr<Action> action;
    action.reset(deform);
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_chunk") {
    std::unique_ptr<Action> action;
    action.reset(new ComputeChunk(tokens, box));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute") {
    std::unique_ptr<Action> action;
    action.reset(new Compute(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "electron_stop") {
    std::unique_ptr<Action> action;
    action.reset(new Electron_Stop(tokens, atom.number_of_atoms, number_of_types));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "add_random_force") {
    std::unique_ptr<Action> action;
    action.reset(new Add_Random_Force(tokens, atom.number_of_atoms));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "add_force") {
    std::unique_ptr<Action> action;
    action.reset(new Add_Force(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "add_spring") {
    std::unique_ptr<Action> action;
    action.reset(new Add_Spring(tokens, group, atom));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "add_efield") {
    std::unique_ptr<Action> action;
    action.reset(new Add_Efield(tokens, group));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "mc") {
    std::unique_ptr<Action> action;
    action.reset(new MC(tokens, group, atom));
    actions_.emplace_back(std::move(action));
  } else if (tokens[0] == "compute_lsqt") {
    std::unique_ptr<Action> action;
    action.reset(new LSQT(tokens));
    actions_.emplace_back(std::move(action));
  } else {
    return false;
  }
  return true;
}

void Measure::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  std::vector<std::string> action_names;
  for (auto& action : actions_) {
    if (action->action_name == "") {
      printf("Dear developer:\n");
      printf("    Please set the action name you developed.\n");
      exit(1);
    }
    // dump_xyz, dump_netcdf, compute_chunk, add_force, add_spring, and add_efield are allowed
    // to be called multiple times; others are not
    if (
      action->action_name != "dump_xyz" && action->action_name != "dump_netcdf" &&
      action->action_name != "compute_chunk" &&
      action->action_name != "add_force" && action->action_name != "add_spring" &&
      action->action_name != "add_efield") {
      for (auto& action_name : action_names) {
        if (action_name == action->action_name) {
          std::cout << "There are multiple " << action->action_name << " keywords within one run.\n";
          exit(1);
        }
      }
    }
    action_names.emplace_back(action->action_name);
  }


  for (auto& action : actions_) {
    action->pre_run(
      number_of_steps,
      time_step,
      integrate,
      group,
      atom,
      box,
      force);
  }
}

void Measure::setup_force(
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : actions_) {
    action->setup_force(time_step, integrate, group, atom, box, force);
  }
}

void Measure::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{

  for (auto& action : actions_) {
    action->post_run(
      atom,
      box,
      integrate,
      number_of_steps,
      time_step,
      temperature);
  }

  actions_.clear();
}

void Measure::end_of_step(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  for (auto& action : actions_) {
    action->end_of_step(
      number_of_steps,
      step,
      fixed_group,
      move_group,
      global_time,
      temperature,
      integrate,
      box,
      group,
      thermo,
      atom,
      force);
  }
}

void Measure::post_integrate1(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : actions_) {
    action->post_integrate1(step, time_step, integrate, group, atom, box, force);
  }
}

void Measure::pre_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : actions_) {
    action->pre_force(step, time_step, integrate, group, atom, box, force);
  }
}

void Measure::post_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  for (auto& action : actions_) {
    action->post_force(step, time_step, integrate, group, atom, box, force);
  }
}
