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
The driver class for the various MC ensembles.
------------------------------------------------------------------------------*/

#include "mc.cuh"
#include "mc_ensemble_canonical.cuh"
#include "mc_ensemble_sgc.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"

void MC::initialize(void)
{
  // todo
}

void MC::finalize(void) { do_mcmd = false; }

void MC::compute(int step, int num_steps, Atom& atom, Box& box, std::vector<Group>& group)
{
  if (do_mcmd) {
    if ((step + 2) % num_steps_md == 0) {
      double temperature =
        temperature_initial + step * (temperature_final - temperature_initial) / num_steps;
      mc_ensemble->compute(step + 2, temperature, atom, box, group, grouping_method, group_id);
    }
  }
}

void MC::parse_group(
  const char** param, int num_param, std::vector<Group>& groups, int num_param_before_group)
{
  if (strcmp(param[num_param_before_group], "group") != 0) {
    PRINT_INPUT_ERROR("invalid option for mc.\n");
  }
  if (!is_valid_int(param[num_param_before_group + 1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method of MCMD should be an integer.\n");
  }
  if (grouping_method < 0) {
    PRINT_INPUT_ERROR("grouping method of MCMD should >= 0.\n");
  }
  if (grouping_method >= groups.size()) {
    PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
  }
  if (!is_valid_int(param[num_param_before_group + 2], &group_id)) {
    PRINT_INPUT_ERROR("group ID of MCMD should be an integer.\n");
  }
  if (group_id < 0) {
    PRINT_INPUT_ERROR("group ID of MCMD should >= 0.\n");
  }
  if (group_id >= groups[grouping_method].number) {
    PRINT_INPUT_ERROR("Group ID should < number of groups.");
  }
}

void MC::check_species_canonical(std::vector<Group>& groups, Atom& atom)
{
  bool has_multi_types = false;
  int type0 = 0;

  if (grouping_method < 0) {
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      if (n == 0) {
        type0 = atom.cpu_type[n];
      } else {
        if (atom.cpu_type[n] != type0) {
          has_multi_types = true;
          break;
        }
      }
    }
  } else {
    for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
      int n =
        groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
      if (k == 0) {
        type0 = atom.cpu_type[n];
      } else {
        if (atom.cpu_type[n] != type0) {
          has_multi_types = true;
          break;
        }
      }
    }
  }

  if (!has_multi_types) {
    PRINT_INPUT_ERROR("Must have more than one atom type for canonical MCMD.");
  }
}

static std::string get_potential_file_name()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string potential_file_name;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "potential") {
        potential_file_name = tokens[1];
        break;
      }
    }
  }

  input_run.close();
  return potential_file_name;
}

static std::vector<std::string> get_atom_symbols_in_nep()
{
  auto potential_file_name = get_potential_file_name();
  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens[0].substr(0, 3) != "nep") {
    PRINT_INPUT_ERROR("MCMD only supports NEP models.");
  }

  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  std::vector<std::string> atom_symbols_in_nep(num_types);
  for (int n = 0; n < num_types; ++n) {
    atom_symbols_in_nep[n] = tokens[2 + n];
  }
  input_potential.close();
  return atom_symbols_in_nep;
}

void MC::check_species_sgc(std::vector<Group>& groups, Atom& atom)
{
  auto atom_symbols_in_nep = get_atom_symbols_in_nep();
  for (int s = 0; s < species.size(); ++s) {
    bool allowed_species = false;
    for (int n = 0; n < atom_symbols_in_nep.size(); ++n) {
      if (species[s] == atom_symbols_in_nep[n]) {
        allowed_species = true;
        types[s] = n;
        break;
      }
    }
    if (!allowed_species) {
      PRINT_INPUT_ERROR("There are listed species not allowed in the NEP model.");
    }
  }

  if (grouping_method < 0) {
    for (int s = 0; s < species.size(); ++s) {
      for (int n = 0; n < atom.number_of_atoms; ++n) {
        if (atom.cpu_atom_symbol[n] == species[s]) {
          ++num_atoms_species[s];
        }
      }
    }
  } else {
    for (int s = 0; s < species.size(); ++s) {
      for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
        int n =
          groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
        if (atom.cpu_atom_symbol[n] == species[s]) {
          ++num_atoms_species[s];
        }
      }
    }
  }

  printf("    the initial system or specified group has:\n");
  bool has_at_least_one_species = false;
  for (int s = 0; s < species.size(); ++s) {
    printf("        %d %s atoms.\n", num_atoms_species[s], species[s].c_str());
    if (num_atoms_species[s] > 0) {
      has_at_least_one_species = true;
    }
  }

  if (!has_at_least_one_species) {
    PRINT_INPUT_ERROR(
      "Must have at least one listed species in the initial model system or specified group.");
  }
}

void MC::parse_mc(const char** param, int num_param, std::vector<Group>& groups, Atom& atom)
{
  if (num_param < 6) {
    PRINT_INPUT_ERROR("mc should have at least 5 parameters.\n");
  }

  int mc_ensemble_type = 0;
  if (strcmp(param[1], "canonical") == 0) {
    printf("Perform canonical MCMD:\n");
    mc_ensemble_type = 0;
  } else if (strcmp(param[1], "sgc") == 0) {
    printf("Perform SGC MCMD:\n");
    mc_ensemble_type = 1;
  } else if (strcmp(param[1], "vcsgc") == 0) {
    printf("Perform VCSGC MCMD:\n");
    mc_ensemble_type = 2;
  } else {
    PRINT_INPUT_ERROR("invalid MC ensemble for MCMD.\n");
  }

  if (!is_valid_int(param[2], &num_steps_md)) {
    PRINT_INPUT_ERROR("number of MD steps for MCMD should be an integer.\n");
  }
  if (num_steps_md <= 0) {
    PRINT_INPUT_ERROR("number of MD steps for MCMD should be positive.\n");
  }

  if (!is_valid_int(param[3], &num_steps_mc)) {
    PRINT_INPUT_ERROR("number of MC steps for MCMD should be an integer.\n");
  }
  if (num_steps_mc <= 0) {
    PRINT_INPUT_ERROR("number of MC steps for MCMD should be positive.\n");
  }

  printf("    after every %d MD steps, do %d MC trials.\n", num_steps_md, num_steps_mc);

  if (!is_valid_real(param[4], &temperature_initial)) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be a number.\n");
  }
  if (temperature_initial <= 0) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be positive.\n");
  }

  if (!is_valid_real(param[5], &temperature_final)) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be a number.\n");
  }
  if (temperature_final <= 0) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be positive.\n");
  }

  printf(
    "    with an initial temperature of %g K and a final temperature of %g K.\n",
    temperature_initial,
    temperature_final);

  if (mc_ensemble_type == 1 || mc_ensemble_type == 2) {
    if (num_param < 7) {
      PRINT_INPUT_ERROR("reading error for num_types in SGC/VCSGC MCMD.\n");
    }
    if (!is_valid_int(param[6], &num_types_mc)) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be an integer.\n");
    }
    if (num_types_mc < 2 || num_types_mc > 4) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be 2 to 4.\n");
    }
    printf("    number of species involved in SGC/VCSGC = %d.\n", num_types_mc);

    if (num_param < (7 + num_types_mc * 2)) {
      PRINT_INPUT_ERROR("not enough (species, mu) or (species, phi) inputs.\n");
    }

    species.resize(num_types_mc);
    mu_or_phi.resize(num_types_mc);
    for (int n = 0; n < num_types_mc; ++n) {
      species[n] = param[7 + n * 2];
      if (!is_valid_real(param[7 + n * 2 + 1], &mu_or_phi[n])) {
        PRINT_INPUT_ERROR("mu or phi should be a number.\n");
      }
      printf("        species = %s, mu/phi= %g\n", species[n].c_str(), mu_or_phi[n]);
    }
  }

  if (mc_ensemble_type == 2) {
    if (num_param < 7 + num_types_mc * 2 + 1) {
      PRINT_INPUT_ERROR("Should have kappa for VCSGC.\n");
    }
    if (!is_valid_real(param[7 + num_types_mc * 2], &kappa)) {
      PRINT_INPUT_ERROR("kappa should be a number.\n");
    }
    if (kappa < 0) {
      PRINT_INPUT_ERROR("kappa should be positive.\n");
    }
    printf("    kappa = %g\n", kappa);
  }

  int num_param_before_group = 6;
  if (mc_ensemble_type == 1) {
    num_param_before_group = 7 + num_types_mc * 2;
  } else if (mc_ensemble_type == 2) {
    num_param_before_group = 8 + num_types_mc * 2;
  }

  if (num_param > num_param_before_group) {
    if (num_param != num_param_before_group + 3) {
      PRINT_INPUT_ERROR("reading error grouping method.\n");
    }
    parse_group(param, num_param, groups, num_param_before_group);
    printf("    only for atoms in group %d of grouping method %d.\n", group_id, grouping_method);
  }

  types.resize(num_types_mc);
  num_atoms_species.resize(num_types_mc, 0);
  if (mc_ensemble_type == 0) {
    check_species_canonical(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_Canonical(param, num_param, num_steps_mc));
  } else if (mc_ensemble_type == 1) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC(
      param, num_param, num_steps_mc, false, species, types, num_atoms_species, mu_or_phi, kappa));
  } else if (mc_ensemble_type == 2) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC(
      param, num_param, num_steps_mc, true, species, types, num_atoms_species, mu_or_phi, kappa));
  }

  do_mcmd = true;
}