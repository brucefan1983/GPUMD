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

void Integrate::initialize(
  double time_step,
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  int& total_steps)
{
  this->total_steps = total_steps;
  int number_of_atoms = atom.number_of_atoms;
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

  // determine the integrator
  switch (type) {
    case EnsembleType::NVE: // NVE
      break;
    case EnsembleType::NVT_BER: // NVT-Berendsen
      break;
    case EnsembleType::NVT_NHC: // NVT-NHC
      break;
    case EnsembleType::NVT_LAN: // NVT-Langevin
      break;
    case EnsembleType::NVT_BDP: // NVT-BDP
      break;
    case EnsembleType::NVT_BAO: // NVT-BAOAB_Langevin
      break;
    case EnsembleType::NVT_QTB: // NVT-QTB
      break;
    case EnsembleType::NPT_BER: // NPT-Berendsen
      break;
    case EnsembleType::NPT_SCR: // NPT-SCR
      break;
    case EnsembleType::MSST: // msst
      break;
    case EnsembleType::TI_SPRING: // ti_spring
      break;
    case EnsembleType::MTTK: // mttk
      break;
    case EnsembleType::WALL_PISTON: // piston
      break;
    case EnsembleType::NPHUG: // nphug
      break;
    case EnsembleType::TI: // ti
      break;
    case EnsembleType::WALL_MIRROR: // mirror
      break;
    case EnsembleType::TI_RS: // ti_rs
      break;
    case EnsembleType::TI_AS: // ti_as
      break;
    case EnsembleType::WALL_HARMONIC:
      break;
    case EnsembleType::TI_LIQUID: // ti_liquid
      break;
    case EnsembleType::NPT_QTB: // npt_qtb
      break;
    case EnsembleType::HEAT_NHC: // heat-NHC
      break;
    // heat with constant power (custom); delta_temperature stores power in eV/fs
    case EnsembleType::HEAT_NHC_POWER:
      break;
    case EnsembleType::HEAT_LAN: // heat-Langevin
      break;
    case EnsembleType::HEAT_BDP: // heat-BDP
      break;
    case EnsembleType::HEAT_TTM: // heat-TTM
      break;
    case EnsembleType::TTM: // pure TTM
      break;
    // Heat-hybrid facilitates the use of both Langevin and Nose-Hoover thermostats
    case EnsembleType::HEAT_HYBRID:
      break;
    case EnsembleType::RPMD: // RPMD
      ensemble.reset(new Ensemble_PIMD(number_of_atoms, number_of_beads, false, atom));
      break;
    case EnsembleType::TRPMD: // TRPMD
      ensemble.reset(new Ensemble_PIMD(number_of_atoms, number_of_beads, true, atom));
      break;
    case EnsembleType::PIMD: // PIMD
      if (num_target_pressure_components == 0) {
        ensemble.reset(new Ensemble_PIMD(
          number_of_atoms,
          number_of_beads,
          temperature_coupling,
          atom,
          use_eco_pimd,
          eco_omega_max_cm1));
      } else {
        ensemble.reset(new Ensemble_PIMD(
          number_of_atoms,
          number_of_beads,
          temperature_coupling,
          num_target_pressure_components,
          target_pressure,
          pressure_coupling,
          atom,
          use_eco_pimd,
          eco_omega_max_cm1,
          use_scr_barostat));
      }
      break;
    default:
      printf("Illegal integrator!\n");
      break;
  }

  ensemble->atom = &atom;
  ensemble->box = &box;
  ensemble->group = &group;
  ensemble->time_step = time_step;
  ensemble->current_step = &this->current_step;
  ensemble->total_steps = &this->total_steps;
  ensemble->thermo = &thermo;
  ensemble->fixed_group = fixed_group;
  ensemble->fixed_grouping_method = fixed_grouping_method;
  ensemble->move_grouping_method = move_grouping_method;
  ensemble->move_group = move_group;
  if (move_group >= 0) {
    for (int i = 0; i < 3; ++i) {
      ensemble->move_velocity[i] = move_velocity[i];
    }
  }
  ensemble->deform_x = deform_x;
  ensemble->deform_y = deform_y;
  ensemble->deform_z = deform_z;
  ensemble->deform_xy = deform_xy;
  ensemble->deform_xz = deform_xz;
  ensemble->deform_yz = deform_yz;
  ensemble->initialize_run(time_step, atom, box, group);
}

void Integrate::finalize()
{
  ensemble.reset();
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
  const double step_over_number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (
    type == EnsembleType::NVE || type == EnsembleType::RPMD ||
    type == EnsembleType::TRPMD) {
    ensemble->temperature = temperature2;
  } else if (is_standard_nvt(type) || is_standard_npt(type) || type == EnsembleType::PIMD) {
    ensemble->temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  ensemble->compute1(time_step, group, box, atom, thermo);
}

void Integrate::compute2(
  const double time_step,
  const double step_over_number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo,
  Force& force)
{
  if (
    type == EnsembleType::NVE || type == EnsembleType::RPMD ||
    type == EnsembleType::TRPMD) {
    ensemble->temperature = temperature2;
  } else if (is_standard_nvt(type) || is_standard_npt(type) || type == EnsembleType::PIMD) {
    ensemble->temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  } else if (type == EnsembleType::TI_LIQUID) {
    ensemble->compute3(time_step, group, box, atom, thermo, force);
    return;
  }

  ensemble->compute2(time_step, group, box, atom, thermo);
}

void Integrate::parse_ensemble(
  const char** param,
  int num_param,
  double time_step,
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo)
{
  if (type != EnsembleType::UNKNOWN) {
    PRINT_INPUT_ERROR("Only one ensemble can be specified before each run.");
  }

  use_eco_pimd = false;
  use_scr_barostat = false;
  eco_omega_max_cm1 = 0.0;
  int pimd_num_param = num_param;

  // 1. Determine the integration method
  if (strcmp(param[1], "nve") == 0) {
    ensemble = std::make_unique<Ensemble_NVE>(num_param);
    type = EnsembleType::NVE;
  } else if (strcmp(param[1], "nvt_ber") == 0 || strcmp(param[1], "npt_ber") == 0) {
    auto ensemble_ber = std::make_unique<Ensemble_BER>(param, num_param, box);
    type = ensemble_ber->type;
    temperature1 = ensemble_ber->get_temperature1();
    temperature2 = ensemble_ber->get_temperature2();
    temperature = temperature1;
    if (type == EnsembleType::NPT_BER) {
      num_target_pressure_components = ensemble_ber->get_num_target_pressure_components();
    }
    ensemble = std::move(ensemble_ber);
  } else if (
    strcmp(param[1], "nvt_nhc") == 0 || strcmp(param[1], "heat_nhc") == 0 ||
    strcmp(param[1], "heat_nhc_power") == 0) {
    auto ensemble_nhc = std::make_unique<Ensemble_NHC>(param, num_param, group);
    type = ensemble_nhc->type;
    temperature = ensemble_nhc->temperature;
    if (type == EnsembleType::NVT_NHC) {
      temperature1 = ensemble_nhc->get_temperature1();
      temperature2 = ensemble_nhc->get_temperature2();
    }
    ensemble = std::move(ensemble_nhc);
  } else if (strcmp(param[1], "nvt_lan") == 0 || strcmp(param[1], "heat_lan") == 0) {
    auto ensemble_lan = std::make_unique<Ensemble_LAN>(param, num_param, group);
    type = ensemble_lan->type;
    temperature = ensemble_lan->temperature;
    if (type == EnsembleType::NVT_LAN) {
      temperature1 = ensemble_lan->get_temperature1();
      temperature2 = ensemble_lan->get_temperature2();
    }
    ensemble = std::move(ensemble_lan);
  } else if (strcmp(param[1], "nvt_bdp") == 0 || strcmp(param[1], "heat_bdp") == 0) {
    auto ensemble_bdp = std::make_unique<Ensemble_BDP>(param, num_param, group);
    type = ensemble_bdp->type;
    temperature = ensemble_bdp->temperature;
    if (type == EnsembleType::NVT_BDP) {
      temperature1 = ensemble_bdp->get_temperature1();
      temperature2 = ensemble_bdp->get_temperature2();
    }
    ensemble = std::move(ensemble_bdp);
  } else if (strcmp(param[1], "nvt_bao") == 0) {
    auto ensemble_bao = std::make_unique<Ensemble_BAO>(param, num_param);
    type = ensemble_bao->type;
    temperature1 = ensemble_bao->get_temperature1();
    temperature2 = ensemble_bao->get_temperature2();
    temperature = temperature1;
    ensemble = std::move(ensemble_bao);
  } else if (strcmp(param[1], "nvt_qtb") == 0) {
    auto ensemble_qtb = std::make_unique<Ensemble_QTB>(param, num_param);
    type = ensemble_qtb->type;
    temperature1 = ensemble_qtb->get_temperature1();
    temperature2 = ensemble_qtb->get_temperature2();
    temperature = ensemble_qtb->temperature;
    ensemble = std::move(ensemble_qtb);
  } else if (strcmp(param[1], "npt_scr") == 0) {
    auto ensemble_scr = std::make_unique<Ensemble_NPT_SCR>(param, num_param, box);
    type = ensemble_scr->type;
    temperature1 = ensemble_scr->get_temperature1();
    temperature2 = ensemble_scr->get_temperature2();
    temperature = ensemble_scr->temperature;
    num_target_pressure_components = ensemble_scr->get_num_target_pressure_components();
    ensemble = std::move(ensemble_scr);
  } else if (
    strcmp(param[1], "nvt_mttk") == 0 || strcmp(param[1], "npt_mttk") == 0 ||
    strcmp(param[1], "nph_mttk") == 0) {
    type = EnsembleType::MTTK;
    Ensemble_MTTK* ptr_temp = new Ensemble_MTTK(param, num_param);
    ensemble.reset(ptr_temp);
    temperature1 = ptr_temp->t_start;
    temperature2 = ptr_temp->t_stop;
  } else if (strcmp(param[1], "npt_qtb") == 0) {
    type = EnsembleType::NPT_QTB;
    Ensemble_NPT_QTB* ptr_temp = new Ensemble_NPT_QTB(param, num_param);
    ensemble.reset(ptr_temp);
    temperature1 = ptr_temp->t_start;
    temperature2 = ptr_temp->t_stop;
  } else if (strcmp(param[1], "heat_ttm") == 0) {
    auto ensemble_ttm =
      std::make_unique<Ensemble_TTM>(param, num_param, atom, box, group);
    type = ensemble_ttm->type;
    temperature = ensemble_ttm->temperature;
    ensemble = std::move(ensemble_ttm);
  } else if (strcmp(param[1], "ttm") == 0) {
    auto ensemble_ttm =
      std::make_unique<Ensemble_TTM>(param, num_param, atom, box, group);
    type = ensemble_ttm->type;
    temperature = ensemble_ttm->temperature;
    temperature1 = 0.0;
    temperature2 = 0.0;
    ensemble = std::move(ensemble_ttm);
  } else if (strcmp(param[1], "heat_hybrid") == 0) {
    auto ensemble_hybrid =
      std::make_unique<Ensemble_Heat_Hybrid>(param, num_param, group);
    type = ensemble_hybrid->type;
    temperature = ensemble_hybrid->temperature;
    ensemble = std::move(ensemble_hybrid);
  } else if (strcmp(param[1], "rpmd") == 0) {
    type = EnsembleType::RPMD;
    if (num_param != 3) {
      PRINT_INPUT_ERROR("ensemble rpmd should have 1 parameter.");
    }
  } else if (strcmp(param[1], "trpmd") == 0) {
    type = EnsembleType::TRPMD;
    if (num_param != 3) {
      PRINT_INPUT_ERROR("ensemble trpmd should have 1 parameter.");
    }
  } else if (strcmp(param[1], "pimd") == 0) {
    type = EnsembleType::PIMD;
  } else if (strcmp(param[1], "pimd_scr") == 0) {
    type = EnsembleType::PIMD;
    use_scr_barostat = true;
  } else if (strcmp(param[1], "msst") == 0) {
    type = EnsembleType::MSST;
    ensemble.reset(new Ensemble_MSST(param, num_param));
  } else if (strcmp(param[1], "ti_spring") == 0) {
    type = EnsembleType::TI_SPRING;
    ensemble.reset(new Ensemble_TI_Spring(param, num_param));
  } else if (strcmp(param[1], "wall_piston") == 0) {
    type = EnsembleType::WALL_PISTON;
    ensemble.reset(new Ensemble_wall_piston(param, num_param));
  } else if (strcmp(param[1], "nphug") == 0) {
    type = EnsembleType::NPHUG;
    ensemble.reset(new Ensemble_NPHug(param, num_param));
  } else if (strcmp(param[1], "ti") == 0) {
    type = EnsembleType::TI;
    ensemble.reset(new Ensemble_TI(param, num_param));
  } else if (strcmp(param[1], "wall_mirror") == 0) {
    type = EnsembleType::WALL_MIRROR;
    ensemble.reset(new Ensemble_wall_mirror(param, num_param));
  } else if (strcmp(param[1], "ti_rs") == 0) {
    type = EnsembleType::TI_RS;
    ensemble.reset(new Ensemble_TI_RS(param, num_param));
  } else if (strcmp(param[1], "ti_as") == 0) {
    type = EnsembleType::TI_AS;
    ensemble.reset(new Ensemble_TI_AS(param, num_param));
  } else if (strcmp(param[1], "wall_harmonic") == 0) {
    type = EnsembleType::WALL_HARMONIC;
    ensemble.reset(new Ensemble_wall_harmonic(param, num_param));
  } else if (strcmp(param[1], "ti_liquid") == 0) {
    type = EnsembleType::TI_LIQUID;
    ensemble.reset(new Ensemble_TI_Liquid(param, num_param));
  } else {
    PRINT_INPUT_ERROR("Invalid ensemble type.");
  }

  // 5. PIMD related
  if (is_pimd(type)) {

    // Optional Eco frequencies are selected by appending
    // "eco omega_max_cm1" to an existing PIMD command.
    if (type == EnsembleType::PIMD) {
      if (num_param >= 8 && strcmp(param[num_param - 2], "eco") == 0) {
        use_eco_pimd = true;
        pimd_num_param = num_param - 2;
        if (!is_valid_real(param[num_param - 1], &eco_omega_max_cm1)) {
          PRINT_INPUT_ERROR("Eco-PIMD omega_max should be a number in cm^-1.");
        }
      }
      if (use_scr_barostat) {
        if (pimd_num_param != 9 && pimd_num_param != 13 && pimd_num_param != 19) {
          PRINT_INPUT_ERROR(
            "ensemble pimd_scr should have 7, 11, or 17 parameters, optionally followed by "
            "eco omega_max_cm1.");
        }
      } else {
        if (
          pimd_num_param != 6 && pimd_num_param != 9 && pimd_num_param != 13 &&
          pimd_num_param != 19) {
          PRINT_INPUT_ERROR(
            "ensemble pimd should have 4, 7, 11, or 17 parameters, optionally followed by "
            "eco omega_max_cm1.");
        }
      }
      if (use_eco_pimd && eco_omega_max_cm1 <= 0.0) {
        PRINT_INPUT_ERROR("Eco-PIMD omega_max should > 0.");
      }
    }

    // number of beads for RPMD, TRPMD, or PIMD
    if (!is_valid_int(param[2], &number_of_beads)) {
      PRINT_INPUT_ERROR("number of beads should be an integer.");
    }
    if (number_of_beads < 2) {
      PRINT_INPUT_ERROR("number of beads should >= 2.");
    }
    if (number_of_beads > MAX_NUM_BEADS) {
      PRINT_INPUT_ERROR("number of beads should <= 128.");
    }
    if (number_of_beads % 2 != 0) {
      PRINT_INPUT_ERROR("number of beads should be an even number.");
    }

    // thermostat and barostat for PIMD
    if (type == EnsembleType::PIMD) {
      // initial temperature
      if (!is_valid_real(param[3], &temperature1)) {
        PRINT_INPUT_ERROR("Initial temperature should be a number.");
      }
      if (temperature1 <= 0.0) {
        PRINT_INPUT_ERROR("Initial temperature should > 0.");
      }
      temperature = temperature1;

      // final temperature
      if (!is_valid_real(param[4], &temperature2)) {
        PRINT_INPUT_ERROR("Final temperature should be a number.");
      }
      if (temperature2 <= 0.0) {
        PRINT_INPUT_ERROR("Final temperature should > 0.");
      }

      // temperature_coupling
      if (!is_valid_real(param[5], &temperature_coupling)) {
        PRINT_INPUT_ERROR("Temperature coupling should be a number.");
      }
      if (temperature_coupling < 1.0) {
        PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
      }

      num_target_pressure_components = 0;

      // pressures:
      if (pimd_num_param >= 9) {
        if (pimd_num_param == 13) {
          for (int i = 0; i < 3; i++) {
            if (!is_valid_real(param[6 + i], &target_pressure[i])) {
              PRINT_INPUT_ERROR("Pressure should be a number.");
            }
          }
          for (int i = 0; i < 3; i++) {
            if (!is_valid_real(param[9 + i], &elastic_modulus[i])) {
              PRINT_INPUT_ERROR("elastic modulus should be a number.");
            }
            if (elastic_modulus[i] <= 0) {
              PRINT_INPUT_ERROR("elastic modulus should > 0.");
            }
          }
          num_target_pressure_components = 3;
          if (
            box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 || box.cpu_h[5] != 0 ||
            box.cpu_h[6] != 0 || box.cpu_h[7] != 0) {
            PRINT_INPUT_ERROR("Cannot use triclinic box with only 3 target pressure components.");
          }
        } else if (pimd_num_param == 9) { // isotropic
          if (!is_valid_real(param[6], &target_pressure[0])) {
            PRINT_INPUT_ERROR("Pressure should be a number.");
          }
          if (!is_valid_real(param[7], &elastic_modulus[0])) {
            PRINT_INPUT_ERROR("elastic modulus should be a number.");
          }
          if (elastic_modulus[0] <= 0) {
            PRINT_INPUT_ERROR("elastic modulus should > 0.");
          }
          num_target_pressure_components = 1;
          if (
            box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 || box.cpu_h[5] != 0 ||
            box.cpu_h[6] != 0 || box.cpu_h[7] != 0) {
            PRINT_INPUT_ERROR("Cannot use triclinic box with only 1 target pressure component.");
          }
          if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
            PRINT_INPUT_ERROR(
              "Cannot use isotropic pressure with non-periodic boundary in any direction.");
          }
        } else { // then must be triclinic box
          for (int i = 0; i < 6; i++) {
            if (!is_valid_real(param[6 + i], &target_pressure[i])) {
              PRINT_INPUT_ERROR("Pressure should be a number.");
            }
          }
          for (int i = 0; i < 6; i++) {
            if (!is_valid_real(param[12 + i], &elastic_modulus[i])) {
              PRINT_INPUT_ERROR("elastic modulus should be a number.");
            }
            if (elastic_modulus[i] <= 0) {
              PRINT_INPUT_ERROR("elastic modulus should > 0.");
            }
          }
          num_target_pressure_components = 6;
          if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
            PRINT_INPUT_ERROR(
              "Cannot use 6 pressure components with non-periodic boundary in any direction.");
          }
        }

        // pressure_coupling:
        int index_pressure_coupling = num_target_pressure_components * 2 + 6;
        if (!is_valid_real(param[index_pressure_coupling], &tau_p)) {
          PRINT_INPUT_ERROR("Pressure coupling should be a number.");
        }
        if (tau_p < 1) {
          PRINT_INPUT_ERROR("Pressure coupling should >= 1.");
        }
        for (int i = 0; i < 6; i++) {
          pressure_coupling[i] = 1.0 / (tau_p * 3.0 * elastic_modulus[i]);
          if (elastic_modulus[i] > 2.0e3) {
            pressure_coupling[i] = 0.0;
          }
        }
      }
    }
  }

  switch (type) {
    case EnsembleType::NVE:
      break;
    case EnsembleType::NVT_BER:
      break;
    case EnsembleType::NVT_NHC:
      break;
    case EnsembleType::NVT_LAN:
      break;
    case EnsembleType::NVT_BDP:
      break;
    case EnsembleType::NVT_BAO:
      break;
    case EnsembleType::NVT_QTB:
      break;
    case EnsembleType::NPT_BER:
      break;
    case EnsembleType::NPT_SCR:
      break;
    case EnsembleType::MSST:
      break;
    case EnsembleType::TI_SPRING:
      break;
    case EnsembleType::MTTK:
      break;
    case EnsembleType::WALL_PISTON:
      break;
    case EnsembleType::NPHUG:
      break;
    case EnsembleType::TI:
      break;
    case EnsembleType::WALL_MIRROR:
      break;
    case EnsembleType::TI_RS:
      break;
    case EnsembleType::TI_AS:
      break;
    case EnsembleType::WALL_HARMONIC:
      break;
    case EnsembleType::TI_LIQUID:
      break;
    case EnsembleType::NPT_QTB: // npt_qtb (self-parsed)
      break;
    case EnsembleType::HEAT_NHC:
      break;
    case EnsembleType::HEAT_NHC_POWER:
      break;
    case EnsembleType::HEAT_LAN:
      break;
    case EnsembleType::HEAT_BDP:
      break;
    case EnsembleType::HEAT_TTM:
      break;
    case EnsembleType::TTM:
      break;
    case EnsembleType::HEAT_HYBRID:
      break;
    case EnsembleType::RPMD:
      printf("Use ring-polymer MD (RPMD) for this run.\n");
      printf("    number of beads is %d.\n", number_of_beads);
      break;
    case EnsembleType::TRPMD:
      printf("Use thermostatted ring-polyer MD (TRPMD) for this run.\n");
      printf("    number of beads is %d.\n", number_of_beads);
      break;
    case EnsembleType::PIMD:
      if (pimd_num_param >= 9) {
        if (use_scr_barostat) {
          printf("Use NPT-PIMD with stochastic cell rescaling for this run.\n");
        } else {
          printf("Use NPT-PIMD for this run.\n");
        }
      } else {
        printf("Use NVT-PIMD for this run.\n");
      }
      printf("    number of beads is %d.\n", number_of_beads);
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      if (pimd_num_param >= 9) {
        if (num_target_pressure_components == 1) {
          printf("    isotropic pressure is %g GPa.\n", target_pressure[0]);
          printf("    bulk modulus is %g GPa.\n", elastic_modulus[0]);
        } else if (num_target_pressure_components == 3) {
          printf("    pressure_xx is %g GPa.\n", target_pressure[0]);
          printf("    pressure_yy is %g GPa.\n", target_pressure[1]);
          printf("    pressure_zz is %g GPa.\n", target_pressure[2]);
          printf("    modulus_xx is %g GPa.\n", elastic_modulus[0]);
          printf("    modulus_yy is %g GPa.\n", elastic_modulus[1]);
          printf("    modulus_zz is %g GPa.\n", elastic_modulus[2]);
        } else if (num_target_pressure_components == 6) {
          printf("    pressure_xx is %g GPa.\n", target_pressure[0]);
          printf("    pressure_yy is %g GPa.\n", target_pressure[1]);
          printf("    pressure_zz is %g GPa.\n", target_pressure[2]);
          printf("    pressure_yz is %g GPa.\n", target_pressure[3]);
          printf("    pressure_xz is %g GPa.\n", target_pressure[4]);
          printf("    pressure_xy is %g GPa.\n", target_pressure[5]);
          printf("    modulus_xx is %g GPa.\n", elastic_modulus[0]);
          printf("    modulus_yy is %g GPa.\n", elastic_modulus[1]);
          printf("    modulus_zz is %g GPa.\n", elastic_modulus[2]);
          printf("    modulus_yz is %g GPa.\n", elastic_modulus[3]);
          printf("    modulus_xz is %g GPa.\n", elastic_modulus[4]);
          printf("    modulus_xy is %g GPa.\n", elastic_modulus[5]);
        }
        printf("    tau_p is %g time_step.\n", tau_p);

        // Change the units of pressure form GPa to that used in the code
        for (int i = 0; i < 6; i++) {
          target_pressure[i] /= PRESSURE_UNIT_CONVERSION;
          pressure_coupling[i] *= PRESSURE_UNIT_CONVERSION;
        }
      }
      break;
    default:
      PRINT_INPUT_ERROR("Invalid ensemble type.");
      break;
  }

  if (type == EnsembleType::PIMD && use_eco_pimd) {
    printf("    use Eco-PIMD internal-mode frequencies.\n");
    printf("    Eco-PIMD omega_max is %g cm^-1.\n", eco_omega_max_cm1);
  }
}

void Integrate::parse_fix(const char** param, int num_param, std::vector<Group>& group)
{
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("Keyword 'fix' should have 1 or 2 parameters.");
  }

  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot use 'fix' without grouping method.");
  }

  if (num_param == 3) {
    // fix grouping_method group_id
    if (!is_valid_int(param[1], &fixed_grouping_method)) {
      PRINT_INPUT_ERROR("Grouping method for 'fix' should be an integer.");
    }
    if (fixed_grouping_method < 0) {
      PRINT_INPUT_ERROR("Grouping method for 'fix' should >= 0.");
    }
    if (fixed_grouping_method >= group.size()) {
      PRINT_INPUT_ERROR("Grouping method for 'fix' should < number of grouping methods.");
    }
    if (!is_valid_int(param[2], &fixed_group)) {
      PRINT_INPUT_ERROR("Fixed group ID should be an integer.");
    }
  } else {
    // fix group_id (default grouping_method = 0)
    fixed_grouping_method = 0;
    if (!is_valid_int(param[1], &fixed_group)) {
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

void Integrate::parse_move(const char** param, int num_param, std::vector<Group>& group)
{
  if (num_param != 5 && num_param != 6) {
    PRINT_INPUT_ERROR("Keyword 'move' should have 4 or 5 parameters.");
  }

  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot use 'move' without grouping method.");
  }

  int vid; // index where vx starts
  if (num_param == 6) {
    // move grouping_method group_id vx vy vz
    if (!is_valid_int(param[1], &move_grouping_method)) {
      PRINT_INPUT_ERROR("Grouping method for 'move' should be an integer.");
    }
    if (move_grouping_method < 0) {
      PRINT_INPUT_ERROR("Grouping method for 'move' should >= 0.");
    }
    if (move_grouping_method >= group.size()) {
      PRINT_INPUT_ERROR("Grouping method for 'move' should < number of grouping methods.");
    }
    if (!is_valid_int(param[2], &move_group)) {
      PRINT_INPUT_ERROR("Moving group ID should be an integer.");
    }
    vid = 3;
  } else {
    // move group_id vx vy vz (default grouping_method = 0)
    move_grouping_method = 0;
    if (!is_valid_int(param[1], &move_group)) {
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

  if (!is_valid_real(param[vid], &move_velocity[0])) {
    PRINT_INPUT_ERROR("Moving velocity in x direction should be a number.");
  }
  if (!is_valid_real(param[vid + 1], &move_velocity[1])) {
    PRINT_INPUT_ERROR("Moving velocity in y direction should be a number.");
  }
  if (!is_valid_real(param[vid + 2], &move_velocity[2])) {
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
