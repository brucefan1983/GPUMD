/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version. GPUMD is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details. You should have
   received a copy of the GNU Lesser General Public License along with GPUMD.  If not, see
   <https://www.gnu.org/licenses/>.
*/

#include "compact_nep.cuh"
#include "utilities/error.cuh"
#include "nep_file.cuh"
#include "nep_model_file.cuh"
#include "nep_spin_model_file.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
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
  float rc_radial;
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
  return NEP_File::is_nep_header(tokens[0]);
}

static std::vector<std::string> get_potential_files(const RunInput& run_input)
{
  std::vector<std::string> files;
  for (const auto& line : run_input.lines()) {
    if (line.tokens.size() >= 2 && line.tokens[0] == "potential") {
      files.push_back(line.tokens[1]);
    }
  }
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
  const std::vector<std::string>& species_full,
  std::vector<int>& required,
  const RunInput& run_input)
{
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
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
}

static std::string get_compact_filename(const int file_index)
{
#ifdef _WIN32
  const int process_id = _getpid();
#else
  const int process_id = getpid();
#endif
  std::string compact_filename = ".gpumd_nep_" +
                                 std::to_string(static_cast<long long>(process_id)) + "_" +
                                 std::to_string(file_index) + ".tmp";
  return compact_filename;
}

static bool make_compact_model_file(
  const std::string& filename,
  const std::vector<std::string>& active_species,
  const int file_index,
  CompactPotentialFile& file)
{
  std::ifstream input(filename);
  NEP_File document;
  NEP_Model_File model;
  std::string error;
  if (!model.read(input, document, error)) {
#ifdef GPUMD_WPE_ENABLED
    return false;
#else
    PRINT_INPUT_ERROR(error.c_str());
#endif
  }
  if (!model.compact(active_species, document, error)) {
    PRINT_INPUT_ERROR(error.c_str());
  }
  const std::string compact_filename = get_compact_filename(file_index);
  std::ofstream output(compact_filename);
  if (!document.write(output, error)) {
    output.close();
    std::remove(compact_filename.c_str());
    PRINT_INPUT_ERROR(error.c_str());
  }
  output.close();
  if (!output) {
    std::remove(compact_filename.c_str());
    PRINT_INPUT_ERROR("Cannot write the compact NEP potential file.");
  }
  file.original = filename;
  file.compact = compact_filename;
  file.rc_radial = *std::max_element(model.rc_radial.begin(), model.rc_radial.end());
  return true;
}

void prepare_compact_nep_files(
  const std::vector<std::string>& atom_symbols, const RunInput& run_input)
{
  if (!cleanup_registered) {
    std::atexit(remove_compact_nep_files);
    cleanup_registered = true;
  }
  remove_compact_nep_files();
  compact_species.clear();

  std::vector<std::string> potential_files = get_potential_files(run_input);
  if (potential_files.size() == 0) {
    PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  }

  std::vector<std::string> first = get_first_line(potential_files.back());
  if (!is_compactable_nep(first) || NEP_Spin_Model_File::is_spin_model(potential_files.back())) {
    return;
  }
  std::vector<std::string> species_full = get_species(first);
  std::vector<int> required(species_full.size(), 0);
  for (int n = 0; n < static_cast<int>(atom_symbols.size()); ++n) {
    add_required_species(species_full, atom_symbols[n], required);
  }
  add_mc_species(species_full, required, run_input);
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

  for (int n = 0; n < static_cast<int>(potential_files.size()); ++n) {
    std::vector<std::string> tokens = get_first_line(potential_files[n]);
    if (!is_compactable_nep(tokens)) {
      continue;
    }
    CompactPotentialFile file;
    if (!make_compact_model_file(potential_files[n], compact_species, n, file)) {
      remove_compact_nep_files();
      compact_species.clear();
      return;
    }
    compact_files.push_back(file);
  }

  printf("Active atom types for this simulation:\n");
  for (int n = 0; n < static_cast<int>(compact_species.size()); ++n) {
    printf("    type %d = %s.\n", n, compact_species[n].c_str());
  }
  printf("    Atom type indices in run.in refer to the active atom types above.\n");
}

void prepare_compact_nep_files(const std::vector<std::string>& atom_symbols)
{
  RunInput run_input("run.in");
  prepare_compact_nep_files(atom_symbols, run_input);
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

// Long-range parameters must retain the cutoff of the full, uncompressed model.
float get_original_nep_radial_cutoff(const std::string& filename, float fallback)
{
  for (const auto& file : compact_files) {
    if (file.compact == filename) {
      return file.rc_radial;
    }
  }
  return fallback;
}

std::string get_first_potential_filename(const RunInput& run_input)
{
  for (const auto& line : run_input.lines()) {
    if (line.tokens.size() >= 2 && line.tokens[0] == "potential") {
      return get_compact_nep_filename(line.tokens[1]);
    }
  }
  return "";
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
