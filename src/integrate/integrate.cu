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

#include "ensemble_bdp.cuh"
#include "ensemble_ber.cuh"
#include "ensemble_lan.cuh"
#include "ensemble_nhc.cuh"
#include "ensemble_nve.cuh"
#include "integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"

void Integrate::initialize(
  const int number_of_atoms, const double time_step, const std::vector<Group>& group)
{
  // determine the integrator
  switch (type) {
    case 0: // NVE
      ensemble.reset(new Ensemble_NVE(type, fixed_group));
      break;
    case 1: // NVT-Berendsen
      ensemble.reset(new Ensemble_BER(type, fixed_group, temperature, temperature_coupling));
      break;
    case 2: // NVT-NHC
      ensemble.reset(new Ensemble_NHC(
        type, fixed_group, number_of_atoms, temperature, temperature_coupling, time_step));
      break;
    case 3: // NVT-Langevin
      ensemble.reset(
        new Ensemble_LAN(type, fixed_group, number_of_atoms, temperature, temperature_coupling));
      break;
    case 4: // NVT-BDP
      ensemble.reset(new Ensemble_BDP(type, fixed_group, temperature, temperature_coupling));
      break;
    case 11: // NPT-Berendsen
      ensemble.reset(new Ensemble_BER(
        type, fixed_group, temperature, temperature_coupling, pressure_x, pressure_y, pressure_z,
        pressure_coupling, deform_x, deform_y, deform_z, deform_rate));
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
    default:
      printf("Illegal integrator!\n");
      break;
  }
}

void Integrate::finalize()
{
  fixed_group = -1; // no group has an index of -1
  deform_x = 0;
  deform_y = 0;
  deform_z = 0;
}

void Integrate::compute1(
  const double time_step,
  const double step_over_number_of_steps,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  if (type >= 1 && type <= 20) {
    ensemble->temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  ensemble->compute1(
    time_step, group, mass, potential_per_atom, force_per_atom, virial_per_atom, box,
    position_per_atom, velocity_per_atom, thermo);
}

void Integrate::compute2(
  const double time_step,
  const double step_over_number_of_steps,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  if (type >= 1 && type <= 20) {
    ensemble->temperature =
      temperature1 + (temperature2 - temperature1) * step_over_number_of_steps;
  }

  ensemble->compute2(
    time_step, group, mass, potential_per_atom, force_per_atom, virial_per_atom, box,
    position_per_atom, velocity_per_atom, thermo);
}

// coding conventions:
// 0:     NVE
// 1-10:  NVT
// 11-20: NPT
// 21-30: heat (NEMD method for heat conductivity)
void Integrate::parse_ensemble(char** param, int num_param, std::vector<Group>& group)
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
  } else if (strcmp(param[1], "npt_ber") == 0) {
    type = 11;
    if (num_param != 9) {
      PRINT_INPUT_ERROR("ensemble npt_ber should have 7 parameters.");
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
    if (temperature_coupling <= 0.0) {
      PRINT_INPUT_ERROR("Temperature coupling should > 0.");
    }
    if (1 == type || 11 == type) // ber
    {
      if (temperature_coupling > 1.0) {
        PRINT_INPUT_ERROR("Temperature coupling should <= 1.");
      }
    } else // nhc, lan, bdp
    {
      if (temperature_coupling < 1.0) {
        PRINT_INPUT_ERROR("Temperature coupling should >= 1.");
      }
    }
  }

  // 3. Pressures and pressure_coupling (NPT)
  double pressure[3];
  if (type >= 11 && type <= 20) {
    // pressures:
    for (int i = 0; i < 3; i++) {
      if (!is_valid_real(param[5 + i], &pressure[i])) {
        PRINT_INPUT_ERROR("Pressure should be a number.");
      }
    }
    // Change the units of pressure form GPa to that used in the code
    pressure_x = pressure[0] / PRESSURE_UNIT_CONVERSION;
    pressure_y = pressure[1] / PRESSURE_UNIT_CONVERSION;
    pressure_z = pressure[2] / PRESSURE_UNIT_CONVERSION;

    // pressure_coupling:
    if (!is_valid_real(param[8], &pressure_coupling)) {
      PRINT_INPUT_ERROR("Pressure coupling should be a number.");
    }
    if (pressure_coupling <= 0.0) {
      PRINT_INPUT_ERROR("Pressure coupling should > 0.");
    }
    if (pressure_coupling > 1) {
      PRINT_INPUT_ERROR("Pressure coupling should <= 1.");
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

  switch (type) {
    case 0:
      printf("Use NVE ensemble for this run.\n");
      break;
    case 1:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Berendsen method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    T_coupling is %g.\n", temperature_coupling);
      break;
    case 2:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Nose-Hoover chain method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    T_coupling is %g.\n", temperature_coupling);
      break;
    case 3:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Langevin method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    T_coupling is %g.\n", temperature_coupling);
      break;
    case 4:
      printf("Use NVT ensemble for this run.\n");
      printf("    choose the Bussi-Donadio-Parrinello method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    T_coupling is %g.\n", temperature_coupling);
      break;
    case 11:
      printf("Use NPT ensemble for this run.\n");
      printf("    choose the Berendsen method.\n");
      printf("    initial temperature is %g K.\n", temperature1);
      printf("    final temperature is %g K.\n", temperature2);
      printf("    T_coupling is %g.\n", temperature_coupling);
      printf("    pressure_x is %g GPa.\n", pressure[0]);
      printf("    pressure_y is %g GPa.\n", pressure[1]);
      printf("    pressure_z is %g GPa.\n", pressure[2]);
      printf("    p_coupling is %g.\n", pressure_coupling);
      break;
    case 21:
      printf("Integrate with heating and cooling for this run.\n");
      printf("    choose the Nose-Hoover chain method.\n");
      printf("    average temperature is %g K.\n", temperature);
      printf("    T_coupling is %g.\n", temperature_coupling);
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
      printf("    T_coupling is %g.\n", temperature_coupling);
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
      printf("    T_coupling is %g.\n", temperature_coupling);
      printf("    delta_T is %g K.\n", delta_temperature);
      printf("    T_hot is %g K.\n", temperature + delta_temperature);
      printf("    T_cold is %g K.\n", temperature - delta_temperature);
      printf("    heat source is group %d in grouping method 0.\n", source);
      printf("    heat sink is group %d in grouping method 0.\n", sink);
      break;
    default:
      PRINT_INPUT_ERROR("Invalid ensemble type.");
      break;
  }
}

void Integrate::parse_fix(char** param, int num_param, std::vector<Group>& group)
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

void Integrate::parse_deform(char** param, int num_param)
{
  printf("Deform the box.\n");

  if (num_param != 5) {
    PRINT_INPUT_ERROR("Keyword 'deform' should have 4 parameters.");
  }

  // strain rate
  if (!is_valid_real(param[1], &deform_rate)) {
    PRINT_INPUT_ERROR("Defrom rate should be a number.");
  }
  printf("    strain rate is %g A / step.\n", deform_rate);

  // direction
  if (!is_valid_int(param[2], &deform_x)) {
    PRINT_INPUT_ERROR("deform_x should be integer.\n");
  }
  if (!is_valid_int(param[3], &deform_y)) {
    PRINT_INPUT_ERROR("deform_y should be integer.\n");
  }
  if (!is_valid_int(param[4], &deform_z)) {
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
