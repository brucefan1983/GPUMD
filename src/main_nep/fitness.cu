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
Get the fitness
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "nep.cuh"
#include "nep_charge.cuh"
#include "nep_charge_vdw.cuh"
#include "nep_vdw.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "tnep.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_parameters.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Number of structures in one mini-batch. The first n_total % num_batches batches take one
// structure more than the rest, so the batches differ in size by at most one.
static int get_batch_size(const int batch_id, const int n_total, const int num_batches)
{
  const int batch_size_minimal = n_total / num_batches;
  const bool is_larger_batch = batch_id + batch_size_minimal * num_batches < n_total;
  return is_larger_batch ? batch_size_minimal + 1 : batch_size_minimal;
}

// One term of a line of ediff.in, with the name in lowercase.
struct EnergyDiffTerm {
  std::string name;
  double coefficient;
};

// One line of ediff.in, a linear combination of the total energies of named structures.
struct EnergyDiffEntry {
  std::vector<EnergyDiffTerm> terms;
  int line_number;
  float weight = 1.0f;
};

static std::string to_lowercase(std::string text)
{
  std::transform(
    text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
  return text;
}

[[noreturn]] static void print_ediff_in_error(const int line_number, const std::string& text)
{
  const std::string message = "ediff.in line " + std::to_string(line_number) + ": " + text;
  PRINT_INPUT_ERROR(message.c_str());
}

// Reads a finite real number that a float holds as zero or as a normal number.
static bool is_valid_float(const std::string& token, double& value)
{
  if (!is_valid_real(token.c_str(), &value) || !std::isfinite(value)) {
    return false;
  }
  const double magnitude = std::fabs(value);
  return magnitude <= FLT_MAX && (magnitude == 0.0 || magnitude >= FLT_MIN);
}

// Reads a nonzero coefficient, either a real number or a fraction p/q of two real numbers.
static bool is_valid_coefficient(const std::string& token, double& value)
{
  const size_t slash = token.find('/');
  if (slash == std::string::npos) {
    return is_valid_float(token, value) && value != 0.0;
  }
  double numerator;
  double denominator;
  if (
    !is_valid_float(token.substr(0, slash), numerator) ||
    !is_valid_float(token.substr(slash + 1), denominator) || denominator == 0.0) {
    return false;
  }
  value = numerator / denominator;
  const double magnitude = std::fabs(value);
  return magnitude >= FLT_MIN && magnitude <= FLT_MAX;
}

// Parses "[+|-] term {(+|-) term} [w=weight]", where a term is "name" or "coefficient*name".
// The coefficient is a real number or a fraction p/q.
static EnergyDiffEntry
parse_ediff_line(const std::vector<std::string>& tokens, const int line_number)
{
  EnergyDiffEntry entry;
  entry.line_number = line_number;
  int k = 0;
  double sign = 1.0;
  if (tokens[0] == "+" || tokens[0] == "-") {
    sign = (tokens[0] == "-") ? -1.0 : 1.0;
    ++k;
  }
  while (true) {
    if (k == (int)tokens.size()) {
      print_ediff_in_error(line_number, "expected a structure after " + tokens[k - 1] + ".");
    }
    const std::string& term = tokens[k++];
    if (term.find('=') != std::string::npos) {
      print_ediff_in_error(line_number, "expected a structure name instead of '" + term + "'.");
    }
    EnergyDiffTerm energy_diff_term;
    energy_diff_term.coefficient = sign;
    const size_t star = term.find('*');
    if (star == std::string::npos) {
      energy_diff_term.name = to_lowercase(term);
    } else {
      const std::string coefficient = term.substr(0, star);
      double value;
      if (!is_valid_coefficient(coefficient, value)) {
        print_ediff_in_error(line_number, "invalid coefficient '" + coefficient + "'.");
      }
      energy_diff_term.coefficient *= value;
      energy_diff_term.name = to_lowercase(term.substr(star + 1));
    }
    if (energy_diff_term.name.empty()) {
      print_ediff_in_error(line_number, "expected a structure name in '" + term + "'.");
    }
    for (const auto& previous_term : entry.terms) {
      if (previous_term.name == energy_diff_term.name) {
        print_ediff_in_error(line_number, energy_diff_term.name + " occurs more than once.");
      }
    }
    entry.terms.push_back(energy_diff_term);
    if (k == (int)tokens.size()) {
      break;
    }
    if (tokens[k] == "+" || tokens[k] == "-") {
      sign = (tokens[k] == "-") ? -1.0 : 1.0;
      ++k;
      continue;
    }
    const std::string weight_string = "w=";
    const bool is_last = k + 1 == (int)tokens.size();
    double bare_weight;
    if (is_last && is_valid_real(tokens[k].c_str(), &bare_weight)) {
      print_ediff_in_error(
        line_number, "a weight is written as w=" + tokens[k] + ", not as " + tokens[k] + ".");
    }
    if (!is_last || tokens[k].substr(0, weight_string.length()) != weight_string) {
      print_ediff_in_error(line_number, "expected + or - before '" + tokens[k] + "'.");
    }
    const std::string weight = tokens[k].substr(weight_string.length());
    double value;
    if (!is_valid_float(weight, value) || value <= 0.0) {
      print_ediff_in_error(
        line_number, "invalid weight '" + weight + "', which should be positive.");
    }
    entry.weight = value;
    break;
  }
  if (entry.terms.size() < 2) {
    print_ediff_in_error(line_number, "a combination needs at least two structures.");
  }
  return entry;
}

static std::vector<EnergyDiffEntry> read_ediff_in(std::ifstream& input)
{
  std::vector<EnergyDiffEntry> entries;
  int line_number = 0;
  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens_without_comments(input);
    ++line_number;
    if (!tokens.empty()) {
      entries.push_back(parse_ediff_line(tokens, line_number));
    }
  }
  return entries;
}

static std::unordered_map<std::string, int>
get_name_to_index(const std::vector<Structure>& structures, const char* xyz_filename)
{
  std::unordered_map<std::string, int> name_to_index;
  for (int nc = 0; nc < (int)structures.size(); ++nc) {
    const std::string& name = structures[nc].name;
    if (name.empty()) {
      continue;
    }
    if (name_to_index.count(name) > 0) {
      const std::string message =
        "the name " + name + " occurs on more than one structure of " + xyz_filename + ".";
      PRINT_INPUT_ERROR(message.c_str());
    }
    name_to_index[name] = nc;
  }
  return name_to_index;
}

// Returns the indices of the structures of each entry whose names all label structures.
static std::vector<std::vector<int>> get_resolved_indices(
  const std::vector<EnergyDiffEntry>& entries, std::unordered_map<std::string, int>& name_to_index)
{
  std::vector<std::vector<int>> resolved_indices;
  for (const auto& entry : entries) {
    std::vector<int> indices;
    for (const auto& term : entry.terms) {
      if (name_to_index.count(term.name) > 0) {
        indices.push_back(name_to_index[term.name]);
      }
    }
    if (indices.size() == entry.terms.size()) {
      resolved_indices.push_back(indices);
    }
  }
  return resolved_indices;
}

/*----------------------------------------------------------------------------80
Reorders the training structures into num_batches batches that keep the structures of each
combination together, and returns the batch sizes. Structures linked through combinations form a
group. The groups, the larger ones first and those of one size by their mean energy per atom, go
one by one to the batch with the fewest structures, which for groups of one structure gives the
batches of read_structures.
Batches left empty are dropped, so num_batches can decrease.
------------------------------------------------------------------------------*/
static std::vector<int> group_structures_by_combination(
  const std::vector<EnergyDiffEntry>& entries,
  std::vector<Structure>& structures,
  const int batch_size,
  int& num_batches)
{
  const int n_total = structures.size();
  std::vector<int> parent(n_total);
  std::iota(parent.begin(), parent.end(), 0);
  auto find_root = [&parent](int n) {
    while (parent[n] != n) {
      parent[n] = parent[parent[n]];
      n = parent[n];
    }
    return n;
  };
  auto name_to_index = get_name_to_index(structures, "train.xyz");
  for (const auto& indices : get_resolved_indices(entries, name_to_index)) {
    for (const int index : indices) {
      parent[find_root(index)] = find_root(indices[0]);
    }
  }

  std::vector<std::vector<int>> groups;
  std::vector<int> root_to_group(n_total, -1);
  for (int n = 0; n < n_total; ++n) {
    const int root = find_root(n);
    if (root_to_group[root] < 0) {
      root_to_group[root] = groups.size();
      groups.emplace_back();
    }
    groups[root_to_group[root]].push_back(n);
  }
  std::vector<double> group_energy(groups.size(), 0.0);
  for (int g = 0; g < (int)groups.size(); ++g) {
    for (const int n : groups[g]) {
      group_energy[g] += structures[n].energy;
    }
    group_energy[g] /= groups[g].size();
  }
  std::vector<int> group_order(groups.size());
  std::iota(group_order.begin(), group_order.end(), 0);
  std::stable_sort(
    group_order.begin(), group_order.end(), [&groups, &group_energy](int g1, int g2) {
      if (groups[g1].size() != groups[g2].size()) {
        return groups[g1].size() > groups[g2].size();
      }
      return group_energy[g1] < group_energy[g2];
    });

  std::vector<std::vector<int>> batches(num_batches);
  for (const int g : group_order) {
    auto smallest = std::min_element(
      batches.begin(), batches.end(), [](const std::vector<int>& b1, const std::vector<int>& b2) {
        return b1.size() < b2.size();
      });
    smallest->insert(smallest->end(), groups[g].begin(), groups[g].end());
  }
  batches.erase(
    std::remove_if(
      batches.begin(), batches.end(), [](const std::vector<int>& batch) { return batch.empty(); }),
    batches.end());
  if ((int)batches.size() < num_batches) {
    num_batches = batches.size();
    printf("Number of batches reduced to %d, since ediff.in links structures.\n", num_batches);
  }
  const int largest_batch =
    std::max_element(
      batches.begin(),
      batches.end(),
      [](const std::vector<int>& b1, const std::vector<int>& b2) { return b1.size() < b2.size(); })
      ->size();
  if (largest_batch > batch_size) {
    printf(
      "Warning: the structures that ediff.in links make a batch of %d structures, which exceeds "
      "the batch size %d.\n",
      largest_batch,
      batch_size);
  }

  std::vector<Structure> structures_grouped;
  structures_grouped.reserve(n_total);
  std::vector<int> batch_sizes;
  for (const auto& batch : batches) {
    for (const int n : batch) {
      structures_grouped.push_back(structures[n]);
    }
    batch_sizes.push_back(batch.size());
  }
  structures.swap(structures_grouped);
  return batch_sizes;
}

// Whether two structures are the same structure for the model: the same types, cell and positions
// to 1e-5 A, the same boundaries, and for model_type 3 the same temperature. Atoms are compared in
// order. The total charge is compared by the caller.
static bool are_the_same_structure(
  const Structure& structure_a, const Structure& structure_b, const int model_type)
{
  const float tolerance = 1.0e-5f;
  if (
    structure_a.num_atom != structure_b.num_atom || structure_a.type != structure_b.type ||
    structure_a.pbc != structure_b.pbc) {
    return false;
  }
  if (model_type == 3 && structure_a.temperature != structure_b.temperature) {
    return false;
  }
  for (int d = 0; d < 9; ++d) {
    if (std::fabs(structure_a.box_original[d] - structure_b.box_original[d]) > tolerance) {
      return false;
    }
  }
  for (int n = 0; n < structure_a.num_atom; ++n) {
    if (
      std::fabs(structure_a.x[n] - structure_b.x[n]) > tolerance ||
      std::fabs(structure_a.y[n] - structure_b.y[n]) > tolerance ||
      std::fabs(structure_a.z[n] - structure_b.z[n]) > tolerance) {
      return false;
    }
  }
  return true;
}

// Returns the combinations of the entries whose names all label structures of one data set,
// split into batches of the given sizes, and marks those entries in is_in_set. A combination has
// to be balanced in the number of atoms of each type, for which any uniform or per-type offset
// of the predicted energies cancels. Its reference is the same combination of the reference
// total energies.
static std::vector<EnergyDiffCombination> resolve_ediff_combinations(
  const std::vector<EnergyDiffEntry>& entries,
  const std::vector<Structure>& structures,
  const Parameters& para,
  const std::vector<int>& batch_sizes,
  const char* xyz_filename,
  std::vector<bool>& is_in_set)
{
  auto name_to_index = get_name_to_index(structures, xyz_filename);

  const int n_total = structures.size();
  std::vector<int> index_to_batch(n_total);
  std::vector<int> index_to_local(n_total);
  int count = 0;
  for (int batch_id = 0; batch_id < (int)batch_sizes.size(); ++batch_id) {
    for (int local = 0; local < batch_sizes[batch_id]; ++local) {
      index_to_batch[count + local] = batch_id;
      index_to_local[count + local] = local;
    }
    count += batch_sizes[batch_id];
  }

  const std::vector<std::string>& elements = para.elements;
  const int num_types = elements.size();
  std::vector<EnergyDiffCombination> combinations;
  for (int k = 0; k < (int)entries.size(); ++k) {
    const EnergyDiffEntry& entry = entries[k];
    const bool is_resolved =
      std::all_of(entry.terms.begin(), entry.terms.end(), [&](const EnergyDiffTerm& term) {
        return name_to_index.count(term.name) > 0;
      });
    if (!is_resolved) {
      continue;
    }
    EnergyDiffCombination combination;
    combination.batch = index_to_batch[name_to_index[entry.terms[0].name]];
    combination.ref_total_eV = 0.0;
    combination.weight = entry.weight;
    std::vector<double> imbalance(num_types, 0.0);
    for (const auto& term : entry.terms) {
      const int index = name_to_index[term.name];
      const Structure& structure = structures[index];
      combination.local.push_back(index_to_local[index]);
      combination.coefficient.push_back(term.coefficient);
      combination.ref_total_eV += term.coefficient * structure.energy_total;
      for (const int type : structure.type) {
        imbalance[type] += term.coefficient;
      }
    }
    for (int i = 0; i < (int)entry.terms.size(); ++i) {
      for (int j = i + 1; j < (int)entry.terms.size(); ++j) {
        const Structure& structure_i = structures[name_to_index[entry.terms[i].name]];
        const Structure& structure_j = structures[name_to_index[entry.terms[j].name]];
        if (!are_the_same_structure(structure_i, structure_j, para.model_type)) {
          continue;
        }
        const std::string names =
          entry.terms[i].name + " and " + entry.terms[j].name + " in " + xyz_filename;
        const bool is_charge_model = para.charge_mode || para.charge_vdw;
        if (is_charge_model && structure_i.charge != structure_j.charge) {
          print_ediff_in_error(
            entry.line_number,
            names + " differ only in charge=, for which a qNEP model predicts no meaningful energy "
                    "difference.");
        }
        print_ediff_in_error(
          entry.line_number,
          names + " have the same geometry, for which the model predicts the same energy.");
      }
    }
    for (int t = 0; t < num_types; ++t) {
      // the atoms left over carry the free energy offset per atom of the population
      if (std::fabs(imbalance[t]) > 1.0e-6) {
        char text[300];
        snprintf(
          text,
          sizeof(text),
          "the combination is not balanced in %s, with %.3g atoms of %s left over.",
          xyz_filename,
          imbalance[t],
          elements[t].c_str());
        std::string message = text;
        if (std::fabs(imbalance[t] - std::round(imbalance[t])) > 1.0e-6) {
          message += " Write a coefficient such as 1/3 as a fraction.";
        }
        print_ediff_in_error(entry.line_number, message);
      }
    }
    combinations.push_back(combination);
    is_in_set[k] = true;
  }
  return combinations;
}

Fitness::Fitness(Parameters& para)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));

  std::vector<Structure> structures_train;
  read_structures(true, para, structures_train);
  num_batches = (structures_train.size() - 1) / para.batch_size + 1;
  printf("Number of devices = %d\n", deviceCount);
  printf("Number of batches = %d\n", num_batches);
  int batch_size_old = para.batch_size;
  para.batch_size = (structures_train.size() - 1) / num_batches + 1;
  if (batch_size_old != para.batch_size) {
    printf("Hello, I changed the batch_size from %d to %d.\n", batch_size_old, para.batch_size);
  }
  std::vector<int> batch_sizes(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    batch_sizes[batch_id] = get_batch_size(batch_id, structures_train.size(), num_batches);
  }

  // The train combinations are resolved before the batches are constructed, which allocate the
  // total energies under para.has_ediff_combinations, and the test combinations once test.xyz
  // has been read.
  std::vector<EnergyDiffEntry> ediff_entries;
  std::vector<bool> is_ediff_entry_in_train;
  std::vector<bool> is_ediff_entry_in_test;
  if (para.prediction == 0) {
    std::ifstream ediff_file("ediff.in");
    if (para.lambda_d > 0.0f) {
      if (!ediff_file.is_open()) {
        PRINT_INPUT_ERROR("lambda_d > 0 requires the file ediff.in.");
      }
      ediff_entries = read_ediff_in(ediff_file);
      is_ediff_entry_in_train.assign(ediff_entries.size(), false);
      is_ediff_entry_in_test.assign(ediff_entries.size(), false);
      // read_structures leaves train.xyz in file order for this grouping
      if (num_batches > 1) {
        batch_sizes = group_structures_by_combination(
          ediff_entries, structures_train, para.batch_size, num_batches);
      }
      ediff_combinations_train = resolve_ediff_combinations(
        ediff_entries, structures_train, para, batch_sizes, "train.xyz", is_ediff_entry_in_train);
      if (ediff_combinations_train.empty()) {
        PRINT_INPUT_ERROR("No combination in ediff.in has all its structures in train.xyz.");
      }
      para.has_ediff_combinations = true;
    } else if (ediff_file.is_open()) {
      printf("ediff.in is ignored because lambda_d = 0.\n");
    }
  }

  train_set.resize(num_batches);
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    train_set[batch_id].resize(deviceCount);
  }
  int count = 0;
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    const int batch_size = batch_sizes[batch_id];
    count += batch_size;
    printf("\nBatch %d:\n", batch_id);
    printf("Number of configurations = %d.\n", batch_size);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing train_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      train_set[batch_id][device_id].construct(
        para, structures_train, count - batch_size, count, device_id);
      print_line_2();
    }
  }

  std::vector<Structure> structures_test;
  has_test_set = read_structures(false, para, structures_test);
  if (para.has_ediff_combinations) {
    if (has_test_set) {
      ediff_combinations_test = resolve_ediff_combinations(
        ediff_entries,
        structures_test,
        para,
        {(int)structures_test.size()},
        "test.xyz",
        is_ediff_entry_in_test);
    }
    int num_in_both = 0;
    int num_skipped = 0;
    int first_skipped = -1;
    for (int k = 0; k < (int)ediff_entries.size(); ++k) {
      if (is_ediff_entry_in_train[k] && is_ediff_entry_in_test[k]) {
        ++num_in_both;
      } else if (!is_ediff_entry_in_train[k] && !is_ediff_entry_in_test[k]) {
        if (num_skipped++ == 0) {
          first_skipped = k;
        }
      }
    }
    printf(
      "ediff.in: %d combinations, %d in train.xyz, %d in test.xyz (%d in both), %d skipped.\n",
      (int)ediff_entries.size(),
      (int)ediff_combinations_train.size(),
      (int)ediff_combinations_test.size(),
      num_in_both,
      num_skipped);
    if (num_skipped > 0) {
      printf(
        "Warning: %d combination(s) of ediff.in skipped, whose structures are neither all in "
        "train.xyz nor all in test.xyz, e.g. line %d.\n",
        num_skipped,
        ediff_entries[first_skipped].line_number);
    }
    if (has_test_set && ediff_combinations_test.empty()) {
      printf("Warning: no combination of ediff.in lies in test.xyz, so rmse_ediff_test is 0.\n");
    }
  }
  if (has_test_set) {
    test_set.resize(deviceCount);
    for (int device_id = 0; device_id < deviceCount; ++device_id) {
      print_line_1();
      printf("Constructing test_set in device  %d.\n", device_id);
      CHECK(gpuSetDevice(device_id));
      test_set[device_id].construct(para, structures_test, 0, structures_test.size(), device_id);
      print_line_2();
    }
  }

  int N = -1;
  int Nc = -1;
  max_NN_radial = -1;
  max_NN_angular = -1;
  if (has_test_set) {
    N = test_set[0].N;
    Nc = test_set[0].Nc;
    max_NN_radial = test_set[0].max_NN_radial;
    max_NN_angular = test_set[0].max_NN_angular;
  }
  for (int n = 0; n < num_batches; ++n) {
    if (train_set[n][0].N > N) {
      N = train_set[n][0].N;
    };
    if (train_set[n][0].Nc > Nc) {
      Nc = train_set[n][0].Nc;
    };

    if (train_set[n][0].max_NN_radial > max_NN_radial) {
      max_NN_radial = train_set[n][0].max_NN_radial;
    }
    if (train_set[n][0].max_NN_angular > max_NN_angular) {
      max_NN_angular = train_set[n][0].max_NN_angular;
    }
  }

  if (para.model_type == 1 || para.model_type == 2) {
    potential.reset(new TNEP(para, N, para.version, deviceCount));
  } else {
    if (para.charge_vdw) {
      potential.reset(new NEP_Charge_VDW(para, N, Nc, para.version, deviceCount));
    } else if (para.charge_mode) {
      potential.reset(new NEP_Charge(para, N, Nc, para.version, deviceCount));
    } else if (para.vdw) {
      potential.reset(new NEP_VDW(para, N, Nc, para.version, deviceCount));
    } else {
      potential.reset(new NEP(para, N, para.version, deviceCount));
    }
  }

  if (para.prediction == 0) {
    fid_loss_out = my_fopen("loss.out", "a");
    fprintf(fid_loss_out, "# format_version 1\n");
    fprintf(fid_loss_out, "# output_interval %d\n", para.output_interval);
    fprintf(fid_loss_out, "# columns generation total L1 L2");
    if (para.model_type == 0 || para.model_type == 3) {
      if (para.charge_mode || para.charge_vdw) {
        fprintf(
          fid_loss_out,
          " rmse_energy_train rmse_force_train rmse_virial_train rmse_charge_train rmse_bec_train"
          " rmse_energy_test rmse_force_test rmse_virial_test rmse_charge_test rmse_bec_test");
      } else {
        fprintf(
          fid_loss_out,
          " rmse_energy_train rmse_force_train rmse_virial_train"
          " rmse_energy_test rmse_force_test rmse_virial_test");
      }
      if (para.has_ediff_combinations) {
        fprintf(fid_loss_out, " rmse_ediff_train rmse_ediff_test");
      }
      fprintf(fid_loss_out, "\n");
    } else if (para.model_type == 1) {
      fprintf(fid_loss_out, " rmse_dipole_train rmse_dipole_test\n");
    } else {
      fprintf(fid_loss_out, " rmse_polarizability_train rmse_polarizability_test\n");
    }
    fflush(fid_loss_out);
  }
}

Fitness::~Fitness()
{
  if (fid_loss_out != NULL) {
    fclose(fid_loss_out);
  }
}

/*----------------------------------------------------------------------------80
Weighted RMSE of the energy combinations whose structures all lie in batch_id of the data set.
The caller must have evaluated the dataset for the parameters of interest. num_combinations
returns the number of contributing combinations, and the RMSE is 0 without any.
------------------------------------------------------------------------------*/
float Fitness::get_rmse_ediff(
  const std::vector<EnergyDiffCombination>& combinations,
  Dataset& dataset,
  const int batch_id,
  const int device_id,
  int& num_combinations)
{
  num_combinations = 0;
  const bool is_any_in_batch = std::any_of(
    combinations.begin(), combinations.end(), [batch_id](const EnergyDiffCombination& combination) {
      return combination.batch == batch_id;
    });
  if (!is_any_in_batch) {
    return 0.0f;
  }
  dataset.compute_total_energies(device_id);
  double sum_sq = 0.0;
  for (const auto& combination : combinations) {
    if (combination.batch != batch_id) {
      continue;
    }
    double difference = -combination.ref_total_eV;
    for (int i = 0; i < (int)combination.local.size(); ++i) {
      difference +=
        combination.coefficient[i] * dataset.total_energy_pred_cpu[combination.local[i]];
    }
    sum_sq += combination.weight * difference * difference;
    ++num_combinations;
  }
  return (num_combinations > 0) ? sqrt(sum_sq / num_combinations) : 0.0f;
}

void Fitness::compute(
  const int generation,
  Parameters& para,
  const float* population,
  float* fitness_energy,
  float* fitness_force,
  float* fitness_virial,
  float* fitness_charge,
  float* fitness_bec,
  float* fitness_ediff)
{
  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  int population_iter = (para.population_size - 1) / deviceCount + 1;

  if (generation == 0) {
    std::vector<float> dummy_solution(para.number_of_variables * deviceCount, para.initial_para);
    for (int n = 0; n < num_batches; ++n) {
      potential->find_force(
        para,
        dummy_solution.data(),
        train_set[n],
        (para.fine_tune || para.import_q_scaler) ? false : true,
        deviceCount);
    }
  } else {
    int batch_id = generation % num_batches;
    for (int n = 0; n < population_iter; ++n) {
      const float* individual = population + deviceCount * n * para.number_of_variables;
      potential->find_force(para, individual, train_set[batch_id], false, deviceCount);
      for (int m = 0; m < deviceCount; ++m) {
        float energy_shift_per_structure_not_used;
        auto rmse_energy_array = train_set[batch_id][m].get_rmse_energy(
          para, energy_shift_per_structure_not_used, true, true, m);
        auto rmse_force_array = train_set[batch_id][m].get_rmse_force(para, true, m);
        auto rmse_virial_array = train_set[batch_id][m].get_rmse_virial(para, true, m);
        auto rmse_charge_array = train_set[batch_id][m].get_rmse_charge(para, m);
        auto rmse_bec_array = train_set[batch_id][m].get_rmse_bec(para, m);

        for (int t = 0; t <= para.num_types; ++t) {
          fitness_energy[deviceCount * n + m + t * para.population_size] =
            para.lambda_e * rmse_energy_array[t];
          fitness_force[deviceCount * n + m + t * para.population_size] =
            para.lambda_f * rmse_force_array[t];
          fitness_virial[deviceCount * n + m + t * para.population_size] =
            para.lambda_v * rmse_virial_array[t];
          fitness_charge[deviceCount * n + m + t * para.population_size] =
            para.lambda_q * rmse_charge_array[t];
          fitness_bec[deviceCount * n + m + t * para.population_size] =
            para.lambda_z * rmse_bec_array[t];
        }

        int num_combinations = 0;
        const float rmse_ediff = get_rmse_ediff(
          ediff_combinations_train, train_set[batch_id][m], batch_id, m, num_combinations);
        fitness_ediff[deviceCount * n + m] = para.lambda_d * rmse_ediff;
      }
    }
  }
}

void Fitness::output(
  bool is_stress,
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int n = 0; n < num_components; ++n) {
      int offset = n * dataset.N + dataset.Na_sum_cpu[nc];
      float data_nc = 0.0f;
      for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
        data_nc += prediction[offset + m];
      }
      if (!is_stress) {
        fprintf(fid, "%g ", data_nc / dataset.Na_cpu[nc]);
      } else {
        fprintf(fid, "%g ", data_nc / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION);
      }
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.Nc + nc];
      if (is_stress) {
        if (ref_value > -1e5) {
          ref_value *= dataset.Na_cpu[nc] / dataset.structures[nc].volume * PRESSURE_UNIT_CONVERSION;
        }
      }
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}

void Fitness::output_atomic(
  int num_components,
  FILE* fid,
  float* prediction,
  float* reference,
  Dataset& dataset)
{
for (int nc = 0; nc < dataset.Nc; ++nc) {
  int offset = dataset.Na_sum_cpu[nc];
  for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
    for (int n = 0; n < num_components; ++n) {
      int index = n * dataset.N + offset + m;
      fprintf(fid, "%g ", prediction[index]);
    }
    for (int n = 0; n < num_components; ++n) {
      float ref_value = reference[n * dataset.N + offset + m];
      if (n == num_components - 1) {
        fprintf(fid, "%g\n", ref_value);
      } else {
        fprintf(fid, "%g ", ref_value);
      }
    }
  }
}
}

void Fitness::write_nep_txt(FILE* fid_nep, Parameters& para, float* elite)
{
  if (para.model_type == 0) { // potential model
    if (!(para.charge_mode || para.charge_vdw)) {
      if (para.version == 4) {
        if (para.enable_zbl) {
          if (para.vdw) {
            fprintf(fid_nep, "nep4_zbl_vdw %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4_zbl %d ", para.num_types);
          }
        } else {
          if (para.vdw) {
            fprintf(fid_nep, "nep4_vdw %d ", para.num_types);
          } else {
            fprintf(fid_nep, "nep4 %d ", para.num_types);
          }
        }
      } 
    } else {
      if (para.charge_vdw) {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl_charge_vdw %d ", para.num_types);
        } else {
          fprintf(fid_nep, "nep4_charge_vdw %d ", para.num_types);
        }
      } else {
        if (para.enable_zbl) {
          fprintf(fid_nep, "nep4_zbl_charge%d %d ", para.charge_mode, para.num_types);
        } else {
          fprintf(fid_nep, "nep4_charge%d %d ", para.charge_mode, para.num_types);
        }
      }
    }
  } else if (para.model_type == 1) { // dipole model
    if (para.version == 4) {
      fprintf(fid_nep, "nep4_dipole %d ", para.num_types);
    }
  } else if (para.model_type == 2) { // polarizability model
    if (para.version == 4) {
      fprintf(fid_nep, "nep4_polarizability %d ", para.num_types);
    }
  } else if (para.model_type == 3) { // temperature model
    if (para.version == 4) {
      if (para.enable_zbl) {
        fprintf(fid_nep, "nep4_zbl_temperature %d ", para.num_types);
      } else {
        fprintf(fid_nep, "nep4_temperature %d ", para.num_types);
      }
    }
  }

  for (int n = 0; n < para.num_types; ++n) {
    fprintf(fid_nep, "%s ", para.elements[n].c_str());
  }
  fprintf(fid_nep, "\n");
  if (para.enable_zbl) {
    if (para.flexible_zbl) {
      fprintf(fid_nep, "zbl 0 0\n");
    } else if (para.use_typewise_cutoff_zbl) {
      fprintf(fid_nep, "zbl %g %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer, para.typewise_cutoff_zbl_factor);
    } else {
      fprintf(fid_nep, "zbl %g %g\n", para.zbl_rc_inner, para.zbl_rc_outer);
    }
  }

  fprintf(fid_nep, "cutoff %g %g ", para.rc_radial[0], para.rc_angular[0]);
  if (para.has_multiple_cutoffs) {
    for (int n = 1; n < para.num_types; ++n) {
      fprintf(fid_nep, "%g %g ", para.rc_radial[n], para.rc_angular[n]);
    }
  }
  fprintf(fid_nep, "%d %d\n", max_NN_radial, max_NN_angular);

  fprintf(fid_nep, "n_max %d %d\n", para.n_max_radial, para.n_max_angular);
  fprintf(fid_nep, "basis_size %d %d\n", para.basis_size_radial, para.basis_size_angular);
  fprintf(fid_nep, "l_max %d %d %d ", para.L_max, (para.has_q_222 ? 2 : 0), para.has_q_1111);
  if (para.has_q_112 || para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_112);
  }
  if (para.has_q_123 || para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_123);
  }
  if (para.has_q_233 || para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_233);
  }
  if (para.has_q_134) {
    fprintf(fid_nep, "%d ", para.has_q_134);
  }
  fprintf(fid_nep, "\n");

  if (para.num_hidden_layers == 2) {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, para.num_neurons2);
  } else {
    fprintf(fid_nep, "ANN %d %d\n", para.num_neurons1, 0);
  }

  std::vector<float> parameters_file(elite, elite + para.number_of_variables);
  const int descriptor_offset = para.number_of_variables_ann * (para.model_type == 2 ? 2 : 1);
#ifdef USE_CJ
  const int num_channels = para.num_types;
#else
  const int num_channels = para.num_types * para.num_types;
#endif
  descriptor_parameters_to_basis_major(
    parameters_file.data(),
    descriptor_offset,
    num_channels,
    para.n_max_radial,
    para.n_max_angular,
    para.basis_size_radial,
    para.basis_size_angular);
  for (int m = 0; m < para.number_of_variables; ++m) {
    fprintf(fid_nep, "%15.7e\n", parameters_file[m]);
  }
  CHECK(gpuSetDevice(0));
  para.q_scaler_gpu[0].copy_to_host(para.q_scaler_cpu.data());
  for (int d = 0; d < para.q_scaler_cpu.size(); ++d) {
    fprintf(fid_nep, "%15.7e\n", para.q_scaler_cpu[d]);
  }
  if (para.flexible_zbl) {
    for (int d = 0; d < 10 * (para.num_types * (para.num_types + 1) / 2); ++d) {
      fprintf(fid_nep, "%15.7e\n", para.zbl_para[d]);
    }
  }
}

void Fitness::get_save_potential_label(Parameters& para, const int generation, std::string& label) {
    if (para.save_potential_format == 1) {
      time_t rawtime;
      time(&rawtime);
      struct tm* timeinfo = localtime(&rawtime);
      char buffer[200];
      strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
      label = std::string(buffer) + std::to_string(generation + 1);
    } else {
      label = "nep_gen" + std::to_string(generation + 1);
    }
}

void Fitness::report_error(
  Parameters& para,
  const int generation,
  const float loss_total,
  const float loss_L1,
  const float loss_L2,
  float* elite)
{
  if (0 == (generation + 1) % para.output_interval) {
    int batch_id = generation % num_batches;
    potential->find_force(para, elite, train_set[batch_id], false, 1);
    float energy_shift_per_structure;
    auto rmse_energy_train_array =
      train_set[batch_id][0].get_rmse_energy(para, energy_shift_per_structure, false, true, 0);
    auto rmse_force_train_array = train_set[batch_id][0].get_rmse_force(para, false, 0);
    auto rmse_virial_train_array = train_set[batch_id][0].get_rmse_virial(para, false, 0);
    auto rmse_charge_train_array = train_set[batch_id][0].get_rmse_charge(para, 0);
    auto rmse_bec_train_array = train_set[batch_id][0].get_rmse_bec(para, 0);

    float rmse_energy_train = rmse_energy_train_array.back();
    float rmse_force_train = rmse_force_train_array.back();
    float rmse_virial_train = rmse_virial_train_array.back();
    float rmse_charge_train = rmse_charge_train_array.back();
    float rmse_bec_train = rmse_bec_train_array.back();

    float rmse_ediff_train = 0.0f;
    if (para.has_ediff_combinations) {
      int num_combinations_not_used = 0;
      rmse_ediff_train = get_rmse_ediff(
        ediff_combinations_train, train_set[batch_id][0], batch_id, 0, num_combinations_not_used);
    }

    // correct the last bias parameter in the NN
    if (para.model_type == 0 || para.model_type == 3) {
      elite[para.number_of_variables_ann - 1] += energy_shift_per_structure;
    }

    float rmse_energy_test = 0.0f;
    float rmse_force_test = 0.0f;
    float rmse_virial_test = 0.0f;
    float rmse_charge_test = 0.0f;
    float rmse_bec_test = 0.0f;
    float rmse_ediff_test = 0.0f;
    if (has_test_set) {
      potential->find_force(para, elite, test_set, false, 1);
      float energy_shift_per_structure_not_used;
      auto rmse_energy_test_array =
        test_set[0].get_rmse_energy(para, energy_shift_per_structure_not_used, false, false, 0);
      auto rmse_force_test_array = test_set[0].get_rmse_force(para, false, 0);
      auto rmse_virial_test_array = test_set[0].get_rmse_virial(para, false, 0);
      auto rmse_charge_test_array = test_set[0].get_rmse_charge(para, 0);
      auto rmse_bec_test_array = test_set[0].get_rmse_bec(para, 0);
      rmse_energy_test = rmse_energy_test_array.back();
      rmse_force_test = rmse_force_test_array.back();
      rmse_virial_test = rmse_virial_test_array.back();
      rmse_charge_test = rmse_charge_test_array.back();
      rmse_bec_test = rmse_bec_test_array.back();
      int num_combinations_not_used = 0;
      rmse_ediff_test =
        get_rmse_ediff(ediff_combinations_test, test_set[0], 0, 0, num_combinations_not_used);
    }

    FILE* fid_nep = my_fopen("nep.txt", "w");
    write_nep_txt(fid_nep, para, elite);
    fclose(fid_nep);

    if (0 == (generation + 1) % para.save_potential) {
      std::string filename;
      get_save_potential_label(para, generation, filename);
      filename += ".txt";

      FILE* fid_nep = my_fopen(filename.c_str(), "w");
      write_nep_txt(fid_nep, para, elite);
      fclose(fid_nep);
    }

    // The ediff columns follow all others, so that the other columns keep their positions.
    auto finish_row = [&](FILE* fid, const char* ediff_format) {
      if (para.has_ediff_combinations) {
        fprintf(fid, ediff_format, rmse_ediff_train, rmse_ediff_test);
      }
      fprintf(fid, "\n");
    };

    if (para.model_type == 0 || para.model_type == 3) {
      if (!(para.charge_mode || para.charge_vdw)) {
        // NEP models
        printf(
          "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test);
        finish_row(stdout, " %-13.5f %-13.5f");
        fprintf(
          fid_loss_out,
          "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f %-13.5f",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test);
        finish_row(fid_loss_out, " %-13.5f %-13.5f");
      } else {
        // qNEP models:
        printf(
          "%-8d %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f "
          "%-9.5f %-9.5f",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_charge_train,
          rmse_bec_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_charge_test,
          rmse_bec_test);
        finish_row(stdout, " %-9.5f %-9.5f");
        fprintf(
          fid_loss_out,
          "%-8d %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f %-9.5f "
          "%-9.5f %-9.5f",
          generation + 1,
          loss_total,
          loss_L1,
          loss_L2,
          rmse_energy_train,
          rmse_force_train,
          rmse_virial_train,
          rmse_charge_train,
          rmse_bec_train,
          rmse_energy_test,
          rmse_force_test,
          rmse_virial_test,
          rmse_charge_test,
          rmse_bec_test);
        finish_row(fid_loss_out, " %-9.5f %-9.5f");
      }
    } else {
      // TNEP models:
      printf(
        "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
      fprintf(
        fid_loss_out,
        "%-8d %-11.5f %-11.5f %-11.5f %-13.5f %-13.5f\n",
        generation + 1,
        loss_total,
        loss_L1,
        loss_L2,
        rmse_virial_train,
        rmse_virial_test);
    }
    fflush(stdout);
    fflush(fid_loss_out);

    if (has_test_set) {
      if (para.model_type == 0 || para.model_type == 3) {
        FILE* fid_force = my_fopen("force_test.out", "w");
        FILE* fid_energy = my_fopen("energy_test.out", "w");
        FILE* fid_virial = my_fopen("virial_test.out", "w");
        FILE* fid_stress = my_fopen("stress_test.out", "w");
        update_energy_force_virial(fid_energy, fid_force, fid_virial, fid_stress, test_set[0]);
        fclose(fid_energy);
        fclose(fid_force);
        fclose(fid_virial);
        fclose(fid_stress);
        if ((para.charge_mode || para.charge_vdw)) {
          FILE* fid_charge = my_fopen("charge_test.out", "w");
          update_charge(fid_charge, test_set[0]);
          fclose(fid_charge);
          if (para.has_bec) {
            FILE* fid_bec = my_fopen("bec_test.out", "w");
            update_bec(fid_bec, test_set[0]);
            fclose(fid_bec);
          }
        }
      } else if (para.model_type == 1) {
        FILE* fid_dipole = my_fopen("dipole_test.out", "w");
        update_dipole(fid_dipole, test_set[0], para.atomic_v);
        fclose(fid_dipole);
      } else if (para.model_type == 2) {
        FILE* fid_polarizability = my_fopen("polarizability_test.out", "w");
        update_polarizability(fid_polarizability, test_set[0], para.atomic_v);
        fclose(fid_polarizability);
      }
    }
  }

  if (0 == (generation + 1) % 1000) {
    predict(para, elite);
  }
}

void Fitness::update_energy_force_virial(
  FILE* fid_energy, FILE* fid_force, FILE* fid_virial, FILE* fid_stress, Dataset& dataset)
{
  dataset.energy.copy_to_host(dataset.energy_cpu.data());
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  dataset.force.copy_to_host(dataset.force_cpu.data());

  for (int nc = 0; nc < dataset.Nc; ++nc) {
    int offset = dataset.Na_sum_cpu[nc];
    for (int m = 0; m < dataset.structures[nc].num_atom; ++m) {
      int n = offset + m;
      fprintf(
        fid_force,
        "%g %g %g %g %g %g\n",
        dataset.force_cpu[n],
        dataset.force_cpu[n + dataset.N],
        dataset.force_cpu[n + dataset.N * 2],
        dataset.force_ref_cpu[n],
        dataset.force_ref_cpu[n + dataset.N],
        dataset.force_ref_cpu[n + dataset.N * 2]);
    }
  }

  output(false, 1, fid_energy, dataset.energy_cpu.data(), dataset.energy_ref_cpu.data(), dataset);

  output(false, 6, fid_virial, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  output(true, 6, fid_stress, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
}

void Fitness::update_charge(FILE* fid_charge, Dataset& dataset)
{
  dataset.charge.copy_to_host(dataset.charge_cpu.data());
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
      fprintf(fid_charge, "%g\n", dataset.charge_cpu[dataset.Na_sum_cpu[nc] + m]);
    }
  }
}

void Fitness::update_bec(FILE* fid_bec, Dataset& dataset)
{
  dataset.bec.copy_to_host(dataset.bec_cpu.data());
  output_atomic(9, fid_bec, dataset.bec_cpu.data(), dataset.bec_ref_cpu.data(), dataset);
}

void Fitness::update_dipole(FILE* fid_dipole, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(3, fid_dipole, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::update_polarizability(FILE* fid_polarizability, Dataset& dataset, bool atomic)
{
  dataset.virial.copy_to_host(dataset.virial_cpu.data());
  if (!atomic) {
    output(false, 6, fid_polarizability, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset);
  } else {
    output_atomic(6, fid_polarizability, dataset.virial_cpu.data(), dataset.avirial_ref_cpu.data(), dataset);
  }
}

void Fitness::predict(Parameters& para, float* elite)
{
  if (para.model_type == 0 || para.model_type == 3) {
    FILE* fid_force = my_fopen("force_train.out", "w");
    FILE* fid_energy = my_fopen("energy_train.out", "w");
    FILE* fid_virial = my_fopen("virial_train.out", "w");
    FILE* fid_stress = my_fopen("stress_train.out", "w");
    FILE* fid_charge = nullptr;
    FILE* fid_bec = nullptr;
    if ((para.charge_mode || para.charge_vdw)) {
      fid_charge = my_fopen("charge_train.out", "w");
      if (para.has_bec) {
        fid_bec = my_fopen("bec_train.out", "w");
      }
    }
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_energy_force_virial(
        fid_energy, fid_force, fid_virial, fid_stress, train_set[batch_id][0]);
      if ((para.charge_mode || para.charge_vdw)) {
        update_charge(fid_charge, train_set[batch_id][0]);
        if (para.has_bec) {
          update_bec(fid_bec, train_set[batch_id][0]);
        }
      }
    }
    fclose(fid_energy);
    fclose(fid_force);
    fclose(fid_virial);
    fclose(fid_stress);
    if ((para.charge_mode || para.charge_vdw)) {
      fclose(fid_charge);
      if (para.has_bec) {
        fclose(fid_bec);
      }
    }
  } else if (para.model_type == 1) {
    FILE* fid_dipole = my_fopen("dipole_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_dipole(fid_dipole, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_dipole);
  } else if (para.model_type == 2) {
    FILE* fid_polarizability = my_fopen("polarizability_train.out", "w");
    for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
      potential->find_force(para, elite, train_set[batch_id], false, 1);
      update_polarizability(fid_polarizability, train_set[batch_id][0], para.atomic_v);
    }
    fclose(fid_polarizability);
  }
}
