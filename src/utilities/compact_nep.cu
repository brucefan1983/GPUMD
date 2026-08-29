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

#include "compact_nep.cuh"
#include "error.cuh"
#include "read_file.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif
#include <vector>

struct CompactPotentialFile
{
  std::string original;
  std::string compact;
};

static std::vector<CompactPotentialFile> compact_files;
static std::vector<std::string> compact_species;
static std::vector<std::string> registered_required_species;
static bool cleanup_registered = false;

static int find_species(const std::vector<std::string>& species, const std::string& symbol)
{
  for (int n = 0; n < static_cast<int>(species.size()); ++n) {
    if (species[n] == symbol) {
      return n;
    }
  }
  return -1;
}

void register_compact_nep_required_species(const std::string& atom_symbol)
{
  if (find_species(registered_required_species, atom_symbol) < 0) {
    registered_required_species.push_back(atom_symbol);
  }
}

static bool is_compactable_nep(const std::vector<std::string>& tokens)
{
  if (tokens.size() < 3) {
    return false;
  }
  return tokens[0].substr(0, 4) == "nep4" || tokens[0].substr(0, 4) == "nep5";
}

static std::vector<std::string> get_potential_files()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("No run.in.");
  }

  std::vector<std::string> files;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 2 && tokens[0] == "potential") {
      files.push_back(tokens[1]);
    }
  }
  input_run.close();
  return files;
}

static std::vector<std::string> get_first_line(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::string error = "Cannot open potential file " + filename + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }
  std::vector<std::string> tokens = get_tokens(input);
  input.close();
  return tokens;
}

static std::vector<std::string> get_species(const std::vector<std::string>& tokens)
{
  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (static_cast<int>(tokens.size()) != num_types + 2) {
    PRINT_INPUT_ERROR("The number of atom symbols in the potential file is incorrect.");
  }
  std::vector<std::string> species(num_types);
  for (int n = 0; n < num_types; ++n) {
    species[n] = tokens[n + 2];
  }
  return species;
}

static void add_required_species(
  const std::vector<std::string>& species_full,
  const std::string& symbol,
  std::vector<int>& required)
{
  int type = find_species(species_full, symbol);
  if (type < 0) {
    std::string error = "The active species " + symbol + " is not supported by the potential.";
    PRINT_INPUT_ERROR(error.c_str());
  }
  required[type] = 1;
}

static void add_mc_species(
  const std::vector<std::string>& species_full, std::vector<int>& required)
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
      if (index >= static_cast<int>(tokens.size())) {
        break;
      }
      add_required_species(species_full, tokens[index], required);
    }
  }
  input_run.close();
}

static int triangular_pair_index(int type1, int type2, int num_types)
{
  if (type1 > type2) {
    int temp = type1;
    type1 = type2;
    type2 = temp;
  }
  return type1 * num_types - type1 * (type1 - 1) / 2 + type2 - type1;
}

static void write_line(std::ofstream& output, const std::vector<std::string>& tokens)
{
  for (int n = 0; n < static_cast<int>(tokens.size()); ++n) {
    if (n > 0) {
      output << " ";
    }
    output << tokens[n];
  }
  output << "\n";
}

static std::string make_compact_file(
  const std::string& filename,
  const std::vector<std::string>& active_species,
  const int file_index)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::string error = "Cannot open potential file " + filename + ".";
    PRINT_INPUT_ERROR(error.c_str());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }
  input.close();
  if (lines.size() < 6) {
    PRINT_INPUT_ERROR("The NEP potential file is incomplete.");
  }

  int line_index = 0;
  std::vector<std::string> first = get_tokens(lines[line_index++]);
  if (!is_compactable_nep(first)) {
    return filename;
  }
  std::vector<std::string> species_full = get_species(first);
  int num_types_full = species_full.size();
  int num_types = active_species.size();
  std::vector<int> active_to_full(num_types);
  for (int n = 0; n < num_types; ++n) {
    active_to_full[n] = find_species(species_full, active_species[n]);
    if (active_to_full[n] < 0) {
      std::string error =
        "The active species " + active_species[n] + " is not supported by the potential.";
      PRINT_INPUT_ERROR(error.c_str());
    }
  }

  bool has_zbl = first[0].find("zbl") != std::string::npos;
  bool is_charge = first[0].find("charge") != std::string::npos;
  bool is_nep5 = first[0].substr(0, 4) == "nep5";
  bool is_polarizability = first[0].find("polarizability") != std::string::npos;
  bool is_temperature = first[0].find("temperature") != std::string::npos;

  std::vector<std::string> zbl_line;
  bool flexible_zbl = false;
  if (has_zbl) {
    zbl_line = get_tokens(lines[line_index++]);
    if (zbl_line.size() < 3) {
      PRINT_INPUT_ERROR("Invalid ZBL line in the NEP potential file.");
    }
    flexible_zbl =
      get_double_from_token(zbl_line[1], __FILE__, __LINE__) == 0.0 &&
      get_double_from_token(zbl_line[2], __FILE__, __LINE__) == 0.0;
  }

  std::vector<std::string> cutoff = get_tokens(lines[line_index++]);
  std::vector<std::string> n_max = get_tokens(lines[line_index++]);
  std::vector<std::string> basis_size = get_tokens(lines[line_index++]);
  std::vector<std::string> l_max = get_tokens(lines[line_index++]);
  std::vector<std::string> ann = get_tokens(lines[line_index++]);

  int n_max_radial = get_int_from_token(n_max[1], __FILE__, __LINE__);
  int n_max_angular = get_int_from_token(n_max[2], __FILE__, __LINE__);
  int basis_size_radial = get_int_from_token(basis_size[1], __FILE__, __LINE__);
  int basis_size_angular = get_int_from_token(basis_size[2], __FILE__, __LINE__);
  int num_L = get_int_from_token(l_max[1], __FILE__, __LINE__);
  for (int n = 2; n < static_cast<int>(l_max.size()); ++n) {
    if (get_int_from_token(l_max[n], __FILE__, __LINE__) != 0) {
      ++num_L;
    }
  }
  int dim = n_max_radial + 1 + (n_max_angular + 1) * num_L;
  if (is_temperature) {
    ++dim;
  }
  int num_neurons = get_int_from_token(ann[1], __FILE__, __LINE__);

  int ann_type_size;
  int ann_global_size;
  int ann_blocks = is_polarizability ? 2 : 1;
  if (is_charge) {
    ann_type_size = (dim + 3) * num_neurons;
    ann_global_size = 2;
  } else if (is_nep5) {
    ann_type_size = (dim + 2) * num_neurons + 1;
    ann_global_size = 1;
  } else {
    ann_type_size = (dim + 2) * num_neurons;
    ann_global_size = 1;
  }

  int ann_block_size_full = num_types_full * ann_type_size + ann_global_size;
  int ann_size_full = ann_blocks * ann_block_size_full;
  int num_types_sq_full = num_types_full * num_types_full;
  int num_radial_basis = (n_max_radial + 1) * (basis_size_radial + 1);
  int num_angular_basis = (n_max_angular + 1) * (basis_size_angular + 1);
  int num_c_radial_full = num_types_sq_full * num_radial_basis;
  int num_c_angular_full = num_types_sq_full * num_angular_basis;
  int parameter_count_full = ann_size_full + num_c_radial_full + num_c_angular_full + dim;
  if (line_index + parameter_count_full > static_cast<int>(lines.size())) {
    PRINT_INPUT_ERROR("Unexpected number of NEP parameters while creating the compact potential.");
  }

#ifdef _WIN32
  const int process_id = _getpid();
#else
  const int process_id = getpid();
#endif
  std::string compact_filename =
    ".gpumd_nep_" + std::to_string(static_cast<long long>(process_id)) + "_" +
    std::to_string(file_index) + ".tmp";
  std::ofstream output(compact_filename);
  if (!output.is_open()) {
    PRINT_INPUT_ERROR("Cannot create the compact NEP potential file.");
  }

  std::vector<std::string> compact_first;
  compact_first.push_back(first[0]);
  compact_first.push_back(std::to_string(num_types));
  for (int n = 0; n < num_types; ++n) {
    compact_first.push_back(active_species[n]);
  }
  write_line(output, compact_first);
  if (has_zbl) {
    write_line(output, zbl_line);
  }

  if (!is_charge && static_cast<int>(cutoff.size()) == num_types_full * 2 + 3) {
    std::vector<std::string> compact_cutoff;
    compact_cutoff.push_back(cutoff[0]);
    for (int n = 0; n < num_types; ++n) {
      int full_type = active_to_full[n];
      compact_cutoff.push_back(cutoff[1 + full_type * 2]);
      compact_cutoff.push_back(cutoff[2 + full_type * 2]);
    }
    compact_cutoff.push_back(cutoff[cutoff.size() - 2]);
    compact_cutoff.push_back(cutoff[cutoff.size() - 1]);
    write_line(output, compact_cutoff);
  } else {
    write_line(output, cutoff);
  }
  write_line(output, n_max);
  write_line(output, basis_size);
  write_line(output, l_max);
  write_line(output, ann);

  int parameter_start = line_index;
  for (int block = 0; block < ann_blocks; ++block) {
    int block_start = parameter_start + block * ann_block_size_full;
    for (int n = 0; n < num_types; ++n) {
      int type_start = block_start + active_to_full[n] * ann_type_size;
      for (int m = 0; m < ann_type_size; ++m) {
        output << lines[type_start + m] << "\n";
      }
    }
    int global_start = block_start + num_types_full * ann_type_size;
    for (int m = 0; m < ann_global_size; ++m) {
      output << lines[global_start + m] << "\n";
    }
  }

  int radial_start = parameter_start + ann_size_full;
  for (int basis = 0; basis < num_radial_basis; ++basis) {
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int pair = active_to_full[t1] * num_types_full + active_to_full[t2];
        output << lines[radial_start + basis * num_types_sq_full + pair] << "\n";
      }
    }
  }

  int angular_start = radial_start + num_c_radial_full;
  for (int basis = 0; basis < num_angular_basis; ++basis) {
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int pair = active_to_full[t1] * num_types_full + active_to_full[t2];
        output << lines[angular_start + basis * num_types_sq_full + pair] << "\n";
      }
    }
  }

  int scaler_start = angular_start + num_c_angular_full;
  for (int n = 0; n < dim; ++n) {
    output << lines[scaler_start + n] << "\n";
  }

  line_index = parameter_start + parameter_count_full;
  if (flexible_zbl) {
    int num_pairs_full = num_types_full * (num_types_full + 1) / 2;
    int zbl_count_full = num_pairs_full * 10;
    if (line_index + zbl_count_full > static_cast<int>(lines.size())) {
      output.close();
      std::remove(compact_filename.c_str());
      PRINT_INPUT_ERROR("Unexpected number of flexible ZBL parameters.");
    }
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = t1; t2 < num_types; ++t2) {
        int pair = triangular_pair_index(active_to_full[t1], active_to_full[t2], num_types_full);
        for (int n = 0; n < 10; ++n) {
          output << lines[line_index + pair * 10 + n] << "\n";
        }
      }
    }
    line_index += zbl_count_full;
  }

  for (; line_index < static_cast<int>(lines.size()); ++line_index) {
    output << lines[line_index] << "\n";
  }
  output.close();
  return compact_filename;
}

void prepare_compact_nep_files(const std::vector<std::string>& atom_symbols)
{
  if (!cleanup_registered) {
    std::atexit(remove_compact_nep_files);
    cleanup_registered = true;
  }
  remove_compact_nep_files();
  compact_species.clear();

  std::vector<std::string> potential_files = get_potential_files();
  if (potential_files.size() == 0) {
    PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  }

  std::vector<std::string> first = get_first_line(potential_files.back());
  if (!is_compactable_nep(first)) {
    return;
  }
  std::vector<std::string> species_full = get_species(first);
  std::vector<int> required(species_full.size(), 0);
  for (int n = 0; n < static_cast<int>(atom_symbols.size()); ++n) {
    add_required_species(species_full, atom_symbols[n], required);
  }
  add_mc_species(species_full, required);
  for (int n = 0; n < static_cast<int>(registered_required_species.size()); ++n) {
    add_required_species(species_full, registered_required_species[n], required);
  }
  for (int n = 0; n < static_cast<int>(species_full.size()); ++n) {
    if (required[n]) {
      compact_species.push_back(species_full[n]);
    }
  }

  if (compact_species.size() == species_full.size()) {
    return;
  }

  printf("Active atom types for this simulation:\n");
  for (int n = 0; n < static_cast<int>(compact_species.size()); ++n) {
    printf("    type %d = %s.\n", n, compact_species[n].c_str());
  }
  printf("    Atom type indices in run.in refer to the active atom types above.\n");

  for (int n = 0; n < static_cast<int>(potential_files.size()); ++n) {
    std::vector<std::string> tokens = get_first_line(potential_files[n]);
    if (!is_compactable_nep(tokens)) {
      continue;
    }
    std::string compact = make_compact_file(potential_files[n], compact_species, n);
    if (compact != potential_files[n]) {
      CompactPotentialFile file;
      file.original = potential_files[n];
      file.compact = compact;
      compact_files.push_back(file);
    }
  }
}

std::string get_compact_nep_filename(const std::string& filename)
{
  for (int n = 0; n < static_cast<int>(compact_files.size()); ++n) {
    if (compact_files[n].original == filename) {
      return compact_files[n].compact;
    }
  }
  return filename;
}

const std::vector<std::string>& get_compact_nep_species() { return compact_species; }

int get_compact_nep_type(const std::string& atom_symbol)
{
  int type = find_species(compact_species, atom_symbol);
  if (type < 0) {
    std::string error = "The active species " + atom_symbol + " is not initialized.";
    PRINT_INPUT_ERROR(error.c_str());
  }
  return type;
}

void remove_compact_nep_files()
{
  for (int n = 0; n < static_cast<int>(compact_files.size()); ++n) {
    std::remove(compact_files[n].compact.c_str());
  }
  compact_files.clear();
}
