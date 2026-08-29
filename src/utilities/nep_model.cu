/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version. GPUMD is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have
   received a copy of the GNU General Public License along with GPUMD.  If not, see
   <http://www.gnu.org/licenses/>.
*/

#include "nep_model.cuh"
#include "error.cuh"
#include "read_file.cuh"
#include <fstream>

static std::vector<std::string> active_species;

static std::vector<std::string> get_first_line_tokens(const std::string& file_potential)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::vector<std::string> tokens = get_tokens(input);
  input.close();
  return tokens;
}

bool is_compactable_nep_model(const std::string& file_potential)
{
  std::vector<std::string> tokens = get_first_line_tokens(file_potential);
  if (tokens.size() == 0) {
    return false;
  }
  return tokens[0].substr(0, 4) == "nep4" || tokens[0].substr(0, 4) == "nep5";
}

std::vector<std::string> get_potential_atom_symbols(const std::string& file_potential)
{
  std::vector<std::string> tokens = get_first_line_tokens(file_potential);
  if (tokens.size() < 3) {
    PRINT_INPUT_ERROR("The first line of the potential file should have at least 3 items.");
  }
  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != num_types + 2) {
    PRINT_INPUT_ERROR("The number of atom symbols in the potential file is incorrect.");
  }
  std::vector<std::string> atom_symbols(num_types);
  for (int n = 0; n < num_types; ++n) {
    atom_symbols[n] = tokens[n + 2];
  }
  return atom_symbols;
}

static int find_species(const std::vector<std::string>& species, const std::string& symbol)
{
  for (int n = 0; n < species.size(); ++n) {
    if (species[n] == symbol) {
      return n;
    }
  }
  return -1;
}

static void add_required_species(
  const std::vector<std::string>& species_full,
  const std::string& symbol,
  std::vector<int>& is_required)
{
  int type = find_species(species_full, symbol);
  if (type < 0) {
    std::string error = "The active species " + symbol + " is not supported by the potential.";
    PRINT_INPUT_ERROR(error.c_str());
  }
  is_required[type] = 1;
}

static void add_mc_species(
  const std::vector<std::string>& species_full, std::vector<int>& is_required)
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("No run.in.");
  }

  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() < 7 || tokens[0] != "mc") {
      continue;
    }
    if (tokens[1] != "sgc" && tokens[1] != "vcsgc") {
      continue;
    }
    int num_types_mc = 0;
    if (!is_valid_int(tokens[6].c_str(), &num_types_mc) || num_types_mc < 1) {
      continue;
    }
    for (int n = 0; n < num_types_mc; ++n) {
      int index = 7 + n * 2;
      if (index >= tokens.size()) {
        break;
      }
      add_required_species(species_full, tokens[index], is_required);
    }
  }
  input_run.close();
}

void initialize_nep_active_species(
  const std::string& file_potential, const std::vector<std::string>& atom_symbols)
{
  std::vector<std::string> species_full = get_potential_atom_symbols(file_potential);
  std::vector<int> is_required(species_full.size(), 0);
  for (int n = 0; n < atom_symbols.size(); ++n) {
    add_required_species(species_full, atom_symbols[n], is_required);
  }
  add_mc_species(species_full, is_required);

  active_species.clear();
  for (int n = 0; n < species_full.size(); ++n) {
    if (is_required[n]) {
      active_species.push_back(species_full[n]);
    }
  }
}

const std::vector<std::string>& get_nep_active_species() { return active_species; }

std::vector<int> get_nep_active_to_full(const std::vector<std::string>& atom_symbols_full)
{
  if (active_species.size() == 0) {
    std::vector<int> active_to_full(atom_symbols_full.size());
    for (int n = 0; n < atom_symbols_full.size(); ++n) {
      active_to_full[n] = n;
    }
    return active_to_full;
  }

  std::vector<int> active_to_full(active_species.size());
  for (int n = 0; n < active_species.size(); ++n) {
    int type = find_species(atom_symbols_full, active_species[n]);
    if (type < 0) {
      std::string error =
        "The active species " + active_species[n] + " is not supported by the potential.";
      PRINT_INPUT_ERROR(error.c_str());
    }
    active_to_full[n] = type;
  }
  return active_to_full;
}

int get_nep_active_type(const std::string& atom_symbol)
{
  int type = find_species(active_species, atom_symbol);
  if (type < 0) {
    std::string error = "The active species " + atom_symbol + " is not initialized.";
    PRINT_INPUT_ERROR(error.c_str());
  }
  return type;
}

std::vector<float> compact_nep_parameters(
  const std::vector<float>& parameters_full,
  int num_types_full,
  const std::vector<int>& active_to_full,
  int ann_type_size,
  int ann_global_size,
  int ann_blocks,
  int n_max_radial,
  int basis_size_radial,
  int n_max_angular,
  int basis_size_angular,
  int tail_size)
{
  int num_types = active_to_full.size();
  int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  int ann_block_size_full = num_types_full * ann_type_size + ann_global_size;
  int ann_size_full = ann_blocks * ann_block_size_full;
  int num_types_sq_full = num_types_full * num_types_full;
  int num_types_sq = num_types * num_types;
  int num_c_radial_full = num_types_sq_full * num_radial_basis;
  int num_c_angular_full = num_types_sq_full * num_angular_basis;
  int num_c_radial = num_types_sq * num_radial_basis;
  int num_c_angular = num_types_sq * num_angular_basis;
  int expected_size = ann_size_full + num_c_radial_full + num_c_angular_full + tail_size;
  if (parameters_full.size() != expected_size) {
    PRINT_INPUT_ERROR("Unexpected number of NEP parameters while compacting the model.");
  }

  std::vector<float> parameters;
  parameters.reserve(
    ann_blocks * (num_types * ann_type_size + ann_global_size) + num_c_radial +
    num_c_angular + tail_size);

  for (int block = 0; block < ann_blocks; ++block) {
    int offset = block * ann_block_size_full;
    for (int t = 0; t < num_types; ++t) {
      int offset_type = offset + active_to_full[t] * ann_type_size;
      for (int n = 0; n < ann_type_size; ++n) {
        parameters.push_back(parameters_full[offset_type + n]);
      }
    }
    int offset_global = offset + num_types_full * ann_type_size;
    for (int n = 0; n < ann_global_size; ++n) {
      parameters.push_back(parameters_full[offset_global + n]);
    }
  }

  int offset_radial = ann_size_full;
  for (int basis = 0; basis < num_radial_basis; ++basis) {
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int type_pair = active_to_full[t1] * num_types_full + active_to_full[t2];
        parameters.push_back(
          parameters_full[offset_radial + basis * num_types_sq_full + type_pair]);
      }
    }
  }

  int offset_angular = offset_radial + num_c_radial_full;
  for (int basis = 0; basis < num_angular_basis; ++basis) {
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int type_pair = active_to_full[t1] * num_types_full + active_to_full[t2];
        parameters.push_back(
          parameters_full[offset_angular + basis * num_types_sq_full + type_pair]);
      }
    }
  }

  int offset_tail = offset_angular + num_c_angular_full;
  for (int n = 0; n < tail_size; ++n) {
    parameters.push_back(parameters_full[offset_tail + n]);
  }

  return parameters;
}

std::vector<float> compact_nep_zbl_parameters(
  const std::vector<float>& parameters_full,
  int num_types_full,
  const std::vector<int>& active_to_full)
{
  int num_pairs_full = num_types_full * (num_types_full + 1) / 2;
  if (parameters_full.size() != num_pairs_full * 10) {
    PRINT_INPUT_ERROR("Unexpected number of flexible ZBL parameters while compacting the model.");
  }

  int num_types = active_to_full.size();
  std::vector<float> parameters(num_types * (num_types + 1) / 2 * 10);
  int count = 0;
  for (int t1 = 0; t1 < num_types; ++t1) {
    for (int t2 = t1; t2 < num_types; ++t2) {
      int full_t1 = active_to_full[t1];
      int full_t2 = active_to_full[t2];
      if (full_t1 > full_t2) {
        int temp = full_t1;
        full_t1 = full_t2;
        full_t2 = temp;
      }
      int pair_full =
        full_t1 * num_types_full - (full_t1 * (full_t1 - 1)) / 2 + (full_t2 - full_t1);
      for (int n = 0; n < 10; ++n) {
        parameters[count++] = parameters_full[pair_full * 10 + n];
      }
    }
  }
  return parameters;
}
