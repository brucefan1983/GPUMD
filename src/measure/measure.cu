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
The driver class dealing with measurement.
------------------------------------------------------------------------------*/

#include "measure.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#define NUM_OF_HEAT_COMPONENTS 5

void Measure::initialize(
  const int number_of_steps,
  const double time_step,
  Box& box,
  std::vector<Group>& group,
  Atom& atom,
  Force& force)
{
  const int number_of_atoms = atom.mass.size();
  const int number_of_potentials = force.potentials.size();
  dos.preprocess(time_step, group, atom.mass);
  sdc.preprocess(number_of_atoms, time_step, group);
  msd.preprocess(number_of_atoms, time_step, group);
  hac.preprocess(number_of_steps);
  viscosity.preprocess(number_of_steps);
  shc.preprocess(number_of_atoms, group);
  compute.preprocess(number_of_atoms, group);
  hnemd.preprocess();
  hnemdec.preprocess(atom.cpu_mass, atom.cpu_type, atom.cpu_type_size);
  modal_analysis.preprocess(atom.cpu_type_size, atom.mass);
  dump_position.preprocess();
  dump_velocity.preprocess();
  dump_restart.preprocess();
  dump_thermo.preprocess();
  dump_force.preprocess(number_of_atoms, group);
  dump_exyz.preprocess(number_of_atoms, 1);
  dump_observer.preprocess(number_of_atoms, number_of_potentials, force);
#ifdef USE_NETCDF
  dump_netcdf.preprocess(number_of_atoms);
#endif
#ifdef USE_PLUMED
  plmd.preprocess(atom.cpu_mass);
#endif
}

void Measure::finalize(
  const int number_of_steps, const double time_step, const double temperature, const double volume)
{
  dump_position.postprocess();
  dump_velocity.postprocess();
  dump_restart.postprocess();
  dump_thermo.postprocess();
  dump_force.postprocess();
  dump_exyz.postprocess();
  dump_observer.postprocess();
  dos.postprocess();
  sdc.postprocess();
  msd.postprocess();
  hac.postprocess(number_of_steps, temperature, time_step, volume);
  viscosity.postprocess(number_of_steps, temperature, time_step, volume);
  shc.postprocess(time_step);
  compute.postprocess();
  hnemd.postprocess();
  hnemdec.postprocess();
  modal_analysis.postprocess();
#ifdef USE_NETCDF
  dump_netcdf.postprocess();
#endif
#ifdef USE_PLUMED
  plmd.postprocess();
#endif

  // TODO: move to the relevant class
  modal_analysis.compute = 1;
  modal_analysis.method = NO_METHOD;
}


void Measure::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const double global_time,
  const double temperature,
  const double energy_transferred[],
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  const int number_of_atoms = atom.cpu_type.size();
  dump_thermo.process(
    step, number_of_atoms, (fixed_group < 0) ? 0 : group[0].cpu_size[fixed_group], box, thermo);
  dump_position.process(
    step, box, group, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
    atom.cpu_position_per_atom);
  dump_velocity.process(step, group, atom.velocity_per_atom, atom.cpu_velocity_per_atom);
  dump_restart.process(
    step, box, group, atom.cpu_atom_symbol, atom.cpu_type, atom.cpu_mass, atom.position_per_atom,
    atom.velocity_per_atom, atom.cpu_position_per_atom, atom.cpu_velocity_per_atom);
  dump_force.process(step, group, atom.force_per_atom);
  dump_exyz.process(
     step, global_time, box, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
     atom.cpu_position_per_atom, atom.velocity_per_atom, atom.cpu_velocity_per_atom, 
     atom.force_per_atom, atom.virial_per_atom, thermo, 0);
  dump_observer.process(step, global_time, box, atom, force, thermo);

  compute.process(
    step, energy_transferred, group, atom.mass, atom.potential_per_atom, atom.force_per_atom,
    atom.velocity_per_atom, atom.virial_per_atom);
  dos.process(step, group, atom.velocity_per_atom);
  sdc.process(step, group, atom.velocity_per_atom);
  msd.process(step, group, atom.unwrapped_position);
  hac.process(
    number_of_steps, step, atom.velocity_per_atom, atom.virial_per_atom, atom.heat_per_atom);
  viscosity.process(number_of_steps, step, atom.mass, atom.velocity_per_atom, atom.virial_per_atom);
  shc.process(step, group, atom.velocity_per_atom, atom.virial_per_atom);
  hnemd.process(
    step, temperature, box.get_volume(), atom.velocity_per_atom, atom.virial_per_atom,
    atom.heat_per_atom);
  hnemdec.process(
    step, temperature, box.get_volume(), atom.velocity_per_atom, atom.virial_per_atom, atom.type,
    atom.mass, atom.potential_per_atom, atom.heat_per_atom);
  modal_analysis.process(
    step, temperature, box.get_volume(), hnemd.fe, atom.velocity_per_atom, atom.virial_per_atom);
#ifdef USE_NETCDF
  dump_netcdf.process(
    step, global_time, box, atom.cpu_type, atom.position_per_atom, atom.cpu_position_per_atom);
#endif
}

// TODO: move to the relevant class
void Measure::parse_compute_gkma(const char** param, int num_param, const int number_of_types)
{
  modal_analysis.compute = 1;
  if (modal_analysis.method == GKMA_METHOD) { // TODO add warning macro
    printf("*******************************************************"
           "WARNING: GKMA method already defined for this run.\n"
           "         Parameters will be overwritten\n"
           "*******************************************************");
  } else if (modal_analysis.method == HNEMA_METHOD) {
    printf("*******************************************************"
           "WARNING: HNEMA method already defined for this run.\n"
           "         GKMA will now run instead.\n"
           "*******************************************************");
  }
  modal_analysis.method = GKMA_METHOD;

  printf("Compute modal heat current using GKMA method.\n");

  /*
   * There is a hidden feature that allows for specification of atom
   * types to included (must be contiguously defined like potentials)
   * -- Works for types only, not groups --
   */

  if (num_param != 6 && num_param != 9) {
    PRINT_INPUT_ERROR("compute_gkma should have 5 parameters.\n");
  }
  if (
    !is_valid_int(param[1], &modal_analysis.sample_interval) ||
    !is_valid_int(param[2], &modal_analysis.first_mode) ||
    !is_valid_int(param[3], &modal_analysis.last_mode)) {
    PRINT_INPUT_ERROR("A parameter for GKMA should be an integer.\n");
  }

  if (strcmp(param[4], "bin_size") == 0) {
    modal_analysis.f_flag = 0;
    if (!is_valid_int(param[5], &modal_analysis.bin_size)) {
      PRINT_INPUT_ERROR("GKMA bin_size must be an integer.\n");
    }
  } else if (strcmp(param[4], "f_bin_size") == 0) {
    modal_analysis.f_flag = 1;
    if (!is_valid_real(param[5], &modal_analysis.f_bin_size)) {
      PRINT_INPUT_ERROR("GKMA f_bin_size must be a real number.\n");
    }
  } else {
    PRINT_INPUT_ERROR("Invalid binning keyword for compute_gkma.\n");
  }

  MODAL_ANALYSIS* g = &modal_analysis;
  // Parameter checking
  if (g->sample_interval < 1 || g->first_mode < 1 || g->last_mode < 1)
    PRINT_INPUT_ERROR("compute_gkma parameters must be positive integers.\n");
  if (g->first_mode > g->last_mode)
    PRINT_INPUT_ERROR("first_mode <= last_mode required.\n");

  printf(
    "    sample_interval is %d.\n"
    "    first_mode is %d.\n"
    "    last_mode is %d.\n",
    g->sample_interval, g->first_mode, g->last_mode);

  if (g->f_flag) {
    if (g->f_bin_size <= 0.0) {
      PRINT_INPUT_ERROR("bin_size must be greater than zero.\n");
    }
    printf(
      "    Bin by frequency.\n"
      "    f_bin_size is %f THz.\n",
      g->f_bin_size);
  } else {
    if (g->bin_size < 1) {
      PRINT_INPUT_ERROR("compute_gkma parameters must be positive integers.\n");
    }
    printf(
      "    Bin by modes.\n"
      "    bin_size is %d bins.\n",
      g->bin_size);
  }

  // Hidden feature implementation
  if (num_param == 9) {
    if (strcmp(param[6], "atom_range") == 0) {
      if (
        !is_valid_int(param[7], &modal_analysis.atom_begin) ||
        !is_valid_int(param[8], &modal_analysis.atom_end)) {
        PRINT_INPUT_ERROR("GKMA atom_begin & atom_end must be integers.\n");
      }
      if (modal_analysis.atom_begin > modal_analysis.atom_end) {
        PRINT_INPUT_ERROR("atom_begin must be less than atom_end.\n");
      }
      if (modal_analysis.atom_begin < 0) {
        PRINT_INPUT_ERROR("atom_begin must be greater than 0.\n");
      }
      if (modal_analysis.atom_end >= number_of_types) {
        PRINT_INPUT_ERROR("atom_end must be greater than 0.\n");
      }
    } else {
      PRINT_INPUT_ERROR("Invalid GKMA keyword.\n");
    }
    printf(
      "    Use select atom range.\n"
      "    Atom types %d to %d.\n",
      modal_analysis.atom_begin, modal_analysis.atom_end);
  } else // default behavior
  {
    modal_analysis.atom_begin = 0;
    modal_analysis.atom_end = number_of_types - 1;
  }
}

// TODO: move to the relevant class
void Measure::parse_compute_hnema(const char** param, int num_param, const int number_of_types)
{
  modal_analysis.compute = 1;
  if (modal_analysis.method == HNEMA_METHOD) {
    printf("*******************************************************\n"
           "WARNING: HNEMA method already defined for this run.\n"
           "         Parameters will be overwritten\n"
           "*******************************************************\n");
  } else if (modal_analysis.method == GKMA_METHOD) {
    printf("*******************************************************\n"
           "WARNING: GKMA method already defined for this run.\n"
           "         HNEMA will now run instead.\n"
           "*******************************************************\n");
  }
  modal_analysis.method = HNEMA_METHOD;

  printf("Compute modal thermal conductivity using HNEMA method.\n");

  /*
   * There is a hidden feature that allows for specification of atom
   * types to included (must be contiguously defined like potentials)
   * -- Works for types only, not groups --
   */

  if (num_param != 10 && num_param != 13) {
    PRINT_INPUT_ERROR("compute_hnema should have 9 parameters.\n");
  }
  if (
    !is_valid_int(param[1], &modal_analysis.sample_interval) ||
    !is_valid_int(param[2], &modal_analysis.output_interval) ||
    !is_valid_int(param[6], &modal_analysis.first_mode) ||
    !is_valid_int(param[7], &modal_analysis.last_mode)) {
    PRINT_INPUT_ERROR("A parameter for HNEMA should be an integer.\n");
  }

  // HNEMD driving force parameters -> Use HNEMD object
  if (!is_valid_real(param[3], &hnemd.fe_x)) {
    PRINT_INPUT_ERROR("fe_x for HNEMD should be a real number.\n");
  }
  printf("    fe_x = %g /A\n", hnemd.fe_x);
  if (!is_valid_real(param[4], &hnemd.fe_y)) {
    PRINT_INPUT_ERROR("fe_y for HNEMD should be a real number.\n");
  }
  printf("    fe_y = %g /A\n", hnemd.fe_y);
  if (!is_valid_real(param[5], &hnemd.fe_z)) {
    PRINT_INPUT_ERROR("fe_z for HNEMD should be a real number.\n");
  }
  printf("    fe_z = %g /A\n", hnemd.fe_z);
  // magnitude of the vector
  hnemd.fe = hnemd.fe_x * hnemd.fe_x;
  hnemd.fe += hnemd.fe_y * hnemd.fe_y;
  hnemd.fe += hnemd.fe_z * hnemd.fe_z;
  hnemd.fe = sqrt(hnemd.fe);

  if (strcmp(param[8], "bin_size") == 0) {
    modal_analysis.f_flag = 0;
    if (!is_valid_int(param[9], &modal_analysis.bin_size)) {
      PRINT_INPUT_ERROR("HNEMA bin_size must be an integer.\n");
    }
  } else if (strcmp(param[8], "f_bin_size") == 0) {
    modal_analysis.f_flag = 1;
    if (!is_valid_real(param[9], &modal_analysis.f_bin_size)) {
      PRINT_INPUT_ERROR("HNEMA f_bin_size must be a real number.\n");
    }
  } else {
    PRINT_INPUT_ERROR("Invalid binning keyword for compute_hnema.\n");
  }

  MODAL_ANALYSIS* h = &modal_analysis;
  // Parameter checking
  if (h->sample_interval < 1 || h->output_interval < 1 || h->first_mode < 1 || h->last_mode < 1)
    PRINT_INPUT_ERROR("compute_hnema parameters must be positive integers.\n");
  if (h->first_mode > h->last_mode)
    PRINT_INPUT_ERROR("first_mode <= last_mode required.\n");
  if (h->output_interval % h->sample_interval != 0)
    PRINT_INPUT_ERROR("sample_interval must divide output_interval an integer\n"
                      " number of times.\n");

  printf(
    "    sample_interval is %d.\n"
    "    output_interval is %d.\n"
    "    first_mode is %d.\n"
    "    last_mode is %d.\n",
    h->sample_interval, h->output_interval, h->first_mode, h->last_mode);

  if (h->f_flag) {
    if (h->f_bin_size <= 0.0) {
      PRINT_INPUT_ERROR("bin_size must be greater than zero.\n");
    }
    printf(
      "    Bin by frequency.\n"
      "    f_bin_size is %f THz.\n",
      h->f_bin_size);
  } else {
    if (h->bin_size < 1) {
      PRINT_INPUT_ERROR("compute_hnema parameters must be positive integers.\n");
    }
    printf(
      "    Bin by modes.\n"
      "    bin_size is %d modes.\n",
      h->bin_size);
  }

  // Hidden feature implementation
  if (num_param == 13) {
    if (strcmp(param[10], "atom_range") == 0) {
      if (
        !is_valid_int(param[11], &modal_analysis.atom_begin) ||
        !is_valid_int(param[12], &modal_analysis.atom_end)) {
        PRINT_INPUT_ERROR("HNEMA atom_begin & atom_end must be integers.\n");
      }
      if (modal_analysis.atom_begin > modal_analysis.atom_end) {
        PRINT_INPUT_ERROR("atom_begin must be less than atom_end.\n");
      }
      if (modal_analysis.atom_begin < 0) {
        PRINT_INPUT_ERROR("atom_begin must be greater than 0.\n");
      }
      if (modal_analysis.atom_end >= number_of_types) {
        PRINT_INPUT_ERROR("atom_end must be greater than 0.\n");
      }
    } else {
      PRINT_INPUT_ERROR("Invalid HNEMA keyword.\n");
    }
    printf(
      "    Use select atom range.\n"
      "    Atom types %d to %d.\n",
      modal_analysis.atom_begin, modal_analysis.atom_end);
  } else // default behavior
  {
    modal_analysis.atom_begin = 0;
    modal_analysis.atom_end = number_of_types - 1;
  }
}
