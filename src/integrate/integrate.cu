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
The driver class for the various integrators.
------------------------------------------------------------------------------*/

#include "ensemble_bao.cuh"
#include "ensemble_bdp.cuh"
#include "ensemble_ber.cuh"
#include "ensemble_lan.cuh"
#include "ensemble_nhc.cuh"
#include "ensemble_npt_scr.cuh"
#include "ensemble_nve.cuh"
#include "ensemble_pimd.cuh"
#include "integrate.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"

void Integrate::initialize(
  const int number_of_atoms, const double time_step, const std::vector<Group>& group, Atom& atom)
{
  if (move_group >= 0) {
    if (fixed_group < 0) {
      PRINT_INPUT_ERROR("It is not allowed to have moving group but no fixed group.");
    }
    if (move_group == fixed_group) {
      PRINT_INPUT_ERROR("The fixed and moving groups cannot be the same.");
    }
    if (type != 1 && type != 2 && type != 4) {
      PRINT_INPUT_ERROR(
        "It is only allowed to use nvt_ber, nvt_nhc, or nvt_bdp with a moving group.");
    }
  }

  // determine the integrator
  switch (type) {
    case 0: // NVE
      ensemble.reset(new Ensemble_NVE(type, fixed_group));
      break;
    case 1: // NVT-Berendsen
      ensemble.reset(new Ensemble_BER(
        type, fixed_group, move_group, move_velocity, temperature, temperature_coupling));
      break;
    case 2: // NVT-NHC
      ensemble.reset(new Ensemble_NHC(
        type, fixed_group, move_group, move_velocity, number_of_atoms, temperature,
        temperature_coupling, time_step));
      break;
    case 3: // NVT-Langevin
      ensemble.reset(
        new Ensemble_LAN(type, fixed_group, number_of_atoms, temperature, temperature_coupling));
      break;
    case 4: // NVT-BDP
      ensemble.reset(new Ensemble_BDP(
        type, fixed_group, move_group, move_velocity, temperature, temperature_coupling));
      break;
    case 5: // NVT-BAOAB_Langevin
      ensemble.reset(
        new Ensemble_BAO(type, fixed_group, number_of_atoms, temperature, temperature_coupling));
      break;
    case 11: // NPT-Berendsen
      ensemble.reset(new Ensemble_BER(
        type, fixed_group, temperature, temperature_coupling, target_pressure,
        num_target_pressure_components, pressure_coupling, deform_x, deform_y, deform_z,
        deform_rate));
      break;
    case 12: // NPT-SCR
      ensemble.reset(new Ensemble_NPT_SCR(
        type, fixed_group, temperature, temperature_coupling, target_pressure,
        num_target_pressure_components, pressure_coupling, deform_x, deform_y, deform_z,
        deform_rate));
      break;
    case 21: // heat-NHC
      ensemble.reset(new Ensemble_NHC(
        type, fixed_group, source, sink, group[0].cpu_size[source], group[0].cpu_size[sink],
        temperature, temperature_coupling, delta_temperature, time_step));
      break;
    case 22: // heat-Langevin
      ensemble.reset(new Ensemble_LAN(
        type, fixed_group, source, sink, group[0].cpu_size[source], group[0].cpu_size[sink],
        group[0].cpu_size_sum[source], group[0].cpu_size_sum[sink], temperature,
        temperature_coupling, delta_temperature));
      break;
    case 23: // heat-BDP
      ensemble.reset(new Ensemble_BDP(
        type, fixed_group, source, sink, temperature, temperature_coupling, delta_temperature));
      break;
    case 31: // NVT-PIMD
      ensemble.reset(new Ensemble_PIMD(
        number_of_atoms, number_of_beads, temperature, temperature_coupling, atom));
      break;
    default:
      printf("Illegal integrator!\n");
      break;
  }
}

void Integrate::finalize()
{
  fixed_group = -1; // no group has an index of -1
  move_group = -1;
  deform_x = 0;
  deform_y = 0;
  deform_z = 0;
}

static __global__ void gpu_copy_position(
  const int number_of_particles,
  const double* g_xi,
  const double* g_yi,
  const double* g_zi,
  double* g_xo,
  double* g_yo,
  double* g_zo)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    g_xo[i] = g_xi[i];
    g_yo[i] = g_yi[i];
    g_zo[i] = g_zi[i];
  }
}

static __global__ void gpu_update_unwrapped_position(
  const int number_of_particles,
  const double* g_xnew,
  const double* g_ynew,
  const double* g_znew,
  const double* g_xold,
  const double* g_yold,
  const double* g_zold,
  double* g_xo,
  double* g_yo,
  double* g_zo)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    g_xo[i] += g_xnew[i] - g_xold[i];
    g_yo[i] += g_ynew[i] - g_yold[i];
    g_zo[i] += g_znew[i] - g_zold[i];
  }
}

void Integrate::compute1(
  const double time_step,
  const double step_over_number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (type == 0) {
    ensemble->temperature = temperature2;
  } else if (type <= 20) {
    ensemble->temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  const int num_atoms = atom.position_per_atom.size() / 3;
  gpu_copy_position<<<(num_atoms - 1) / 128 + 1, 128>>>(
    num_atoms, atom.position_per_atom.data(), atom.position_per_atom.data() + num_atoms,
    atom.position_per_atom.data() + num_atoms * 2, atom.position_temp.data(),
    atom.position_temp.data() + num_atoms, atom.position_temp.data() + num_atoms * 2);
  CUDA_CHECK_KERNEL

  ensemble->compute1(time_step, group, box, atom, thermo);

  gpu_update_unwrapped_position<<<(num_atoms - 1) / 128 + 1, 128>>>(
    num_atoms, atom.position_per_atom.data(), atom.position_per_atom.data() + num_atoms,
    atom.position_per_atom.data() + num_atoms * 2, atom.position_temp.data(),
    atom.position_temp.data() + num_atoms, atom.position_temp.data() + num_atoms * 2,
    atom.unwrapped_position.data(), atom.unwrapped_position.data() + num_atoms,
    atom.unwrapped_position.data() + num_atoms * 2);
  CUDA_CHECK_KERNEL
}

void Integrate::compute2(
  const double time_step,
  const double step_over_number_of_steps,
  const std::vector<Group>& group,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& thermo)
{
  if (type == 0) {
    ensemble->temperature = temperature2;
  } else if (type <= 20) {
    ensemble->temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  ensemble->compute2(time_step, group, box, atom, thermo);
}

// coding conventions:
// 0:     NVE
// 1-10:  NVT
// 11-20: NPT
// 21-30: heat (NEMD method for heat conductivity)
void Integrate::parse_ensemble(
  Box& box, const char** param, int num_param, std::vector<Group>& group)
{
  // 1. Determine the integration method
  if (strcmp(param[1], "nve") == 0) {
    type = 0;
    if (num_param != 2) {
      PRINT_INPUT_ERROR("ensemble nve should have 0 parameter.");
    }
  } else if (strcmp(param[1], "nvt_ber") == 0) {
    type = 1;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_ber should have 3 parameters.");
    }
  } else if (strcmp(param[1], "nvt_nhc") == 0) {
    type = 2;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_nhc should have 3 parameters.");
    }
  } else if (strcmp(param[1], "nvt_lan") == 0) {
    type = 3;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_lan should have 3 parameters.");
    }
  } else if (strcmp(param[1], "nvt_bdp") == 0) {
    type = 4;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_bdp should have 3 parameters.");
    }
  } else if (strcmp(param[1], "nvt_bao") == 0) {
    type = 5;
    if (num_param != 5) {
      PRINT_INPUT_ERROR("ensemble nvt_bao should have 3 parameters.");
    }
  } else if (strcmp(param[1], "npt_ber") == 0) {
    type = 11;
    if (num_param != 18 && num_param != 12 && num_param != 8) {
      PRINT_INPUT_ERROR("ensemble npt_ber should have 6, 10, or 16 parameters.");
    }
  } else if (strcmp(param[1], "npt_scr") == 0) {
    type = 12;
    if (num_param != 18 && num_param != 12 && num_param != 8) {
      PRINT_INPUT_ERROR("ensemble npt_scr should have 6, 10, or 16 parameters.");
    }
  } else if (strcmp(param[1], "heat_nhc") == 0) {
    type = 21;
    if (num_param != 7) {
      PRINT_INPUT_ERROR("ensemble heat_nhc should have 5 parameters.");
    }
  } else if (strcmp(param[1], "heat_lan") == 0) {
    type = 22;
    if (num_param != 7) {
      PRINT_INPUT_ERROR("ensemble heat_lan should have 5 parameters.");
    }
  } else if (strcmp(param[1], "heat_bdp") == 0) {
    type = 23;
    if (num_param != 7) {
      PRINT_INPUT_ERROR("ensemble heat_bdp should have 5 parameters.");
    }
  } else if (strcmp(param[1], "nvt_pimd") == 0) {
    type = 31;
    if (num_param != 6) {
      PRINT_INPUT_ERROR("ensemble nvt_pimd should have 4 parameters.");
    }
  } else {
    PRINT_INPUT_ERROR("Invalid ensemble type.");
  }

  // 2. Temperatures and temperature_coupling (NVT and NPT)
  if (type >= 1 && type <= 20) {
    // initial temperature
    if (!is_valid_real(param[2], &temperature1)) {
      PRINT_INPUT_ERROR("Initial temperature should be a number.");
    }
    if (temperature1 <= 0.0) {
      PRINT_INPUT_ERROR("Initial temperature should > 0.");
    }

    // final temperature
    if (!is_valid_real(param[3], &temperature2)) {
      PRINT_INPUT_ERROR("Final temperature should be a number.");
    }
    if (temperature2 <= 0.0) {
      PRINT_INPUT_ERROR("Final temperature should > 0.");
    }

    // The current temperature is the initial temperature
    temperature = temperature1;

    // temperature_coupling
    if (!is_valid_real(param[4], &temperature_coupling)) {
      PRINT_INPUT_ERROR("Temperature coupling should be a number.");
    }
    if (temperature_coupling < 1.0) {
      if (type == 1 || type == 11) {
        PRINT_INPUT_ERROR(
          "Temperature coupling should >= 1. \n(We have changed the convention for this "
          "input starting from GPUMD-V3.0; See the manual for details.)");
      } else {
        PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
      }
    }
  }

  // 3. Pressures and pressure_coupling (NPT)
  if (type >= 11 && type <= 20) {
    // pressures:
    if (num_param == 12) {
      for (int i = 0; i < 3; i++) {
        if (!is_valid_real(param[5 + i], &target_pressure[i])) {
          PRINT_INPUT_ERROR("Pressure should be a number.");
        }
      }
      for (int i = 0; i < 3; i++) {
        if (!is_valid_real(param[8 + i], &elastic_modulus[i])) {
          PRINT_INPUT_ERROR("elastic modulus should be a number.");
        }
        if (elastic_modulus[i] <= 0) {
          PRINT_INPUT_ERROR("elastic modulus should > 0.");
        }
      }
      num_target_pressure_components = 3;
      if (box.triclinic == 1) {
        PRINT_INPUT_ERROR("Cannot use triclinic box with only 3 target pressure components.");
      }
    } else if (num_param == 8) { // isotropic
      if (!is_valid_real(param[5], &target_pressure[0])) {
        PRINT_INPUT_ERROR("Pressure should be a number.");
      }
      if (!is_valid_real(param[6], &elastic_modulus[0])) {
        PRINT_INPUT_ERROR("elastic modulus should be a number.");
      }
      if (elastic_modulus[0] <= 0) {
        PRINT_INPUT_ERROR("elastic modulus should > 0.");
      }
      num_target_pressure_components = 1;
      if (box.triclinic == 1) {
        PRINT_INPUT_ERROR("Cannot use triclinic box with only 1 target pressure component.");
      }
      if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
        PRINT_INPUT_ERROR(
          "Cannot use isotropic pressure with non-periodic boundary in any direction.");
      }
    } else { // then must be triclinic box
      for (int i = 0; i < 6; i++) {
        if (!is_valid_real(param[5 + i], &target_pressure[i])) {
          PRINT_INPUT_ERROR("Pressure should be a number.");
        }
      }
      for (int i = 0; i < 6; i++) {
        if (!is_valid_real(param[11 + i], &elastic_modulus[i])) {
          PRINT_INPUT_ERROR("elastic modulus should be a number.");
        }
        if (elastic_modulus[i] <= 0) {
          PRINT_INPUT_ERROR("elastic modulus should > 0.");
        }
      }
      num_target_pressure_components = 6;
      if (box.triclinic == 0) {
        PRINT_INPUT_ERROR("Must use triclinic box with 6 target pressure components.");
      }
      if (box.pbc_x == 0 || box.pbc_y == 0 || box.pbc_z == 0) {
        PRINT_INPUT_ERROR(
          "Cannot use 6 pressure components with non-periodic boundary in any direction.");
      }
    }

    // pressure_coupling:
    int index_pressure_coupling = num_target_pressure_components * 2 + 5;
    if (!is_valid_real(param[index_pressure_coupling], &tau_p)) {
      PRINT_INPUT_ERROR("Pressure coupling should be a number.");
    }
    if (tau_p < 1) {
      if (type == 11) {
        PRINT_INPUT_ERROR(
          "Pressure coupling should >= 1. \n(We have changed the convention for this "
          "input starting from GPUMD-V3.0; See the manual for details.)");
      } else {
        PRINT_INPUT_ERROR("Pressure coupling should >= 1.");
      }
    }
    for (int i = 0; i < 6; i++) {
      pressure_coupling[i] = 1.0 / (tau_p * 3.0 * elastic_modulus[i]);
    }
  }

  // 4. heating and cooling wiht fixed temperatures
  if (type >= 21 && type <= 30) {
    // temperature
    if (!is_valid_real(param[2], &temperature)) {
      PRINT_INPUT_ERROR("Temperature should be a number.");
    }
    if (temperature <= 0.0) {
      PRINT_INPUT_ERROR("Temperature should > 0.");
    }

    // temperature_coupling
    if (!is_valid_real(param[3], &temperature_coupling)) {
      PRINT_INPUT_ERROR("Temperature coupling should be a number.");
    }
    if (temperature_coupling < 1.0) {
      PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
    }

    // temperature difference
    if (!is_valid_real(param[4], &delta_temperature)) {
      PRINT_INPUT_ERROR("Temperature difference should be a number.");
    }
    if (delta_temperature >= temperature || delta_temperature <= -temperature) {
      PRINT_INPUT_ERROR("|Temperature difference| is too large.");
    }

    // group labels of heat source and sink
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

  // 5. NVT-PIMD
  if (type == 31) {
    // temperature
    if (!is_valid_real(param[2], &temperature)) {
      PRINT_INPUT_ERROR("temperature should be a number.");
    }
    if (temperature <= 0.0) {
      PRINT_INPUT_ERROR("temperature should > 0.");
    }

    // temperature_coupling for the physical particles
    if (!is_valid_real(param[3], &temperature_coupling)) {
      PRINT_INPUT_ERROR("Temperature coupling should be a number.");
    }
    if (temperature_coupling < 1.0) {
      PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
    }

    // temperature_coupling for the beads
    if (!is_valid_real(param[4], &temperature_coupling_beads)) {
      PRINT_INPUT_ERROR("Temperature coupling should be a number.");
    }
    if (temperature_coupling_beads < 1.0) {
      PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
    }

    // number of beads
    if (!is_valid_int(param[5], &number_of_beads)) {
      PRINT_INPUT_ERROR("number of beads should be an integer.");
    }
    if (number_of_beads < 2 || number_of_beads > 128) {
      PRINT_INPUT_ERROR("number of beads should be >= 2 and <= 128.");
    }
    if (number_of_beads % 2 != 0) {
      PRINT_INPUT_ERROR("number of beads should be an even number.");
    }
  }

  switch (type) {
    case 0:
      printf("Use NVE ensemble for this run.\n");
      break;
    case 1:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Berendsen method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      break;
    case 2:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Nose-Hoover chain method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      break;
    case 3:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Langevin method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      break;
    case 4:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Bussi-Donadio-Parrinello method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      break;
    case 5:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the BAOAB Langevin method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      break;
    case 11:
      printf("Use NPT ensemble for this run.\n");
      printf("    choose the Berendsen method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step\n", temperature_coupling);
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
      break;
    case 12:
      printf("Use NPT ensemble for this run.\n");
      printf("    choose the SCR method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
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
      break;
    case 21:
      printf("Integrate with heating and cooling for this run.\n");
      printf("    choose the Nose-Hoover chain method.\n");
      printf("    average temperature is %g K.\n", temperature);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      printf("    delta_T is %g K.\n", delta_temperature);
      printf("    T_hot is %g K.\n", temperature + delta_temperature);
      printf("    T_cold is %g K.\n", temperature - delta_temperature);
      printf("    heat source is group %d in grouping method 0.\n", source);
      printf("    heat sink is group %d in grouping method 0.\n", sink);
      break;
    case 22:
      printf("Integrate with heating and cooling for this run.\n");
      printf("    choose the Langevin method.\n");
      printf("    average temperature is %g K.\n", temperature);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      printf("    delta_T is %g K.\n", delta_temperature);
      printf("    T_hot is %g K.\n", temperature + delta_temperature);
      printf("    T_cold is %g K.\n", temperature - delta_temperature);
      printf("    heat source is group %d in grouping method 0.\n", source);
      printf("    heat sink is group %d in grouping method 0.\n", sink);
      break;
    case 23:
      printf("Integrate with heating and cooling for this run.\n");
      printf("    choose the Bussi-Donadio-Parrinello method.\n");
      printf("    average temperature is %g K.\n", temperature);
      printf("    tau_T is %g time_step.\n", temperature_coupling);
      printf("    delta_T is %g K.\n", delta_temperature);
      printf("    T_hot is %g K.\n", temperature + delta_temperature);
      printf("    T_cold is %g K.\n", temperature - delta_temperature);
      printf("    heat source is group %d in grouping method 0.\n", source);
      printf("    heat sink is group %d in grouping method 0.\n", sink);
      break;
    case 31:
      printf("Use NVT-PIMD ensemble for this run.\n");
      printf("    temperature is %g K.\n", temperature);
      printf("    physical coupling is %g time_step.\n", temperature_coupling);
      printf("    internal coupling is %g time_step.\n", temperature_coupling_beads);
      printf("    number of beads is %d.\n", number_of_beads);
      break;
    default:
      PRINT_INPUT_ERROR("Invalid ensemble type.");
      break;
  }
}

void Integrate::parse_fix(const char** param, int num_param, std::vector<Group>& group)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("Keyword 'fix' should have 1 parameter.");
  }

  if (!is_valid_int(param[1], &fixed_group)) {
    PRINT_INPUT_ERROR("Fixed group ID should be an integer.");
  }

  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot use 'fix' without grouping method.");
  }

  if (fixed_group < 0) {
    PRINT_INPUT_ERROR("Fixed group ID should >= 0.");
  }

  if (fixed_group >= group[0].number) {
    PRINT_INPUT_ERROR("Fixed group ID should < number of groups.");
  }

  printf("Group %d in grouping method 0 will be fixed.\n", fixed_group);
}

void Integrate::parse_move(const char** param, int num_param, std::vector<Group>& group)
{
  if (num_param != 5) {
    PRINT_INPUT_ERROR("Keyword 'move' should have 4 parameters.");
  }

  if (!is_valid_int(param[1], &move_group)) {
    PRINT_INPUT_ERROR("Moving group ID should be an integer.");
  }

  if (group.size() < 1) {
    PRINT_INPUT_ERROR("Cannot use 'move' without grouping method.");
  }

  if (move_group < 0) {
    PRINT_INPUT_ERROR("Moving group ID should >= 0.");
  }

  if (move_group >= group[0].number) {
    PRINT_INPUT_ERROR("Moving group ID should < number of groups.");
  }

  if (!is_valid_real(param[2], &move_velocity[0])) {
    PRINT_INPUT_ERROR("Moving velocity in x direction should be a number.");
  }
  if (!is_valid_real(param[3], &move_velocity[1])) {
    PRINT_INPUT_ERROR("Moving velocity in y direction should be a number.");
  }
  if (!is_valid_real(param[4], &move_velocity[2])) {
    PRINT_INPUT_ERROR("Moving velocity in z direction should be a number.");
  }

  printf(
    "Group %d in grouping method 0 will move with velocity vector (%g, %g, %g) A/fs.\n", move_group,
    move_velocity[0], move_velocity[1], move_velocity[2]);

  for (int d = 0; d < 3; ++d) {
    move_velocity[d] *= TIME_UNIT_CONVERSION; // natural to A/fs
  }
}

void Integrate::parse_deform(const char** param, int num_param)
{
  printf("Deform the box.\n");

  if (num_param != 5 && num_param != 7) {
    PRINT_INPUT_ERROR("Keyword 'deform' should have 4 or 6 parameters.");
  }

  // strain rate
  if (!is_valid_real(param[1], &deform_rate[0])) {
    PRINT_INPUT_ERROR("Defrom rate should be a number.");
  }

  int offset = 0;
  if (num_param == 5) {
    deform_rate[1] = deform_rate[0];
    deform_rate[2] = deform_rate[0];
    printf("    strain rate is %g A / step.\n", deform_rate[0]);
  } else {
    offset = 2;
    if (!is_valid_real(param[2], &deform_rate[1])) {
      PRINT_INPUT_ERROR("Defrom rate should be a number.");
    }
    if (!is_valid_real(param[3], &deform_rate[2])) {
      PRINT_INPUT_ERROR("Defrom rate should be a number.");
    }
    printf(
      "    strain rates are (%g, %g, %g) A / step.\n", deform_rate[0], deform_rate[1],
      deform_rate[2]);
  }

  // direction
  if (!is_valid_int(param[2 + offset], &deform_x)) {
    PRINT_INPUT_ERROR("deform_x should be integer.\n");
  }
  if (!is_valid_int(param[3 + offset], &deform_y)) {
    PRINT_INPUT_ERROR("deform_y should be integer.\n");
  }
  if (!is_valid_int(param[4 + offset], &deform_z)) {
    PRINT_INPUT_ERROR("deform_z should be integer.\n");
  }

  if (deform_x) {
    printf("    apply strain in x direction.\n");
  }
  if (deform_y) {
    printf("    apply strain in y direction.\n");
  }
  if (deform_z) {
    printf("    apply strain in z direction.\n");
  }
}
