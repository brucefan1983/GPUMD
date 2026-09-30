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
The energy-difference loss over the linear combinations of named structures in ediff.in
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "energy_difference.cuh"
#include "parameters.cuh"
#include "structure.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

// Whether ediff.in can refer to a structure by this name, which is the case unless it is empty,
// begins with #, + or -, or contains whitespace or any of * / = " ' { }.
static bool is_valid_structure_name(const std::string& name)
{
  return !name.empty() && name.front() != '#' && name.front() != '+' && name.front() != '-' &&
         name.find_first_of("*/=\"'{} \t") == std::string::npos;
}

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
    if (!is_valid_structure_name(energy_diff_term.name)) {
      const char first = energy_diff_term.name.front();
      if (first == '+' || first == '-') {
        print_ediff_in_error(
          line_number, "write the sign of a term as a field of its own, as in '- name'.");
      }
      print_ediff_in_error(
        line_number, "'" + energy_diff_term.name + "' is not a valid structure name.");
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
    char* end = nullptr;
    std::strtod(tokens[k].c_str(), &end);
    const bool is_number = end != tokens[k].c_str() && *end == '\0';
    if (is_last && is_number) {
      print_ediff_in_error(
        line_number, "a weight is written as w=" + tokens[k] + ", not as " + tokens[k] + ".");
    }
    if (!is_last || tokens[k].substr(0, weight_string.length()) != weight_string) {
      if (tokens[k].size() > 1 && (tokens[k].front() == '+' || tokens[k].front() == '-')) {
        print_ediff_in_error(
          line_number, "write the sign of a term as a field of its own, as in '- name'.");
      }
      print_ediff_in_error(line_number, "expected + or - before '" + tokens[k] + "'.");
    }
    const std::string weight = tokens[k].substr(weight_string.length());
    double value;
    if (!is_valid_float(weight, value) || value <= 0.0) {
      print_ediff_in_error(
        line_number,
        "invalid weight '" + weight +
          "', which should be a positive number within the range of "
          "float.");
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

// Whether two structures are the same structure for the model: the same types, positions to
// 1e-5 A, boundaries, cell for periodic boundaries, and for model_type 3 temperature. Atoms are
// compared in order. The total charge is compared by the caller.
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
  for (int d = 0; structure_a.pbc && d < 9; ++d) {
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
          std::fabs(imbalance[t]),
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

/*----------------------------------------------------------------------------80
Weighted RMSE of the energy combinations whose structures all lie in batch_id of the data set.
The caller must have evaluated the dataset for the parameters of interest. The RMSE is 0 without
any such combination.
------------------------------------------------------------------------------*/
static float get_rmse(
  const std::vector<EnergyDiffCombination>& combinations,
  Dataset& dataset,
  const int batch_id,
  const int device_id)
{
  const bool is_any_in_batch = std::any_of(
    combinations.begin(), combinations.end(), [batch_id](const EnergyDiffCombination& combination) {
      return combination.batch == batch_id;
    });
  if (!is_any_in_batch) {
    return 0.0f;
  }
  dataset.compute_total_energies(device_id);
  double sum_sq = 0.0;
  int num_combinations = 0;
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
  return sqrt(sum_sq / num_combinations);
}

std::string EnergyDifference::read_structure_name(
  const std::vector<std::string>& tokens, const std::string& xyz_filename, const int line_number)
{
  const std::string location = xyz_filename + " line " + std::to_string(line_number);
  std::string structure_name;
  // a name= inside the quoted or bracketed value of another field is part of that value;
  // closing is the character that ends the value open at the start of a token, or 0
  const std::string opening_delimiters = "\"'{[";
  const std::string closing_delimiters = "\"'}]";
  char closing = 0;
  for (int n = 0; n < tokens.size(); ++n) {
    const char closing_at_start = closing;
    for (int c = 0; c < tokens[n].size(); ++c) {
      const char character = tokens[n][c];
      const size_t kind = opening_delimiters.find(character);
      if (character == '\\') {
        ++c;
      } else if (closing == 0 && kind != std::string::npos) {
        closing = closing_delimiters[kind];
      } else if (character == closing) {
        closing = 0;
      }
    }
    const std::string name_string = "name=";
    if (closing_at_start != 0 || tokens[n].substr(0, name_string.length()) != name_string) {
      continue;
    }
    if (!structure_name.empty()) {
      PRINT_INPUT_ERROR((location + ": more than one name= field.").c_str());
    }
    std::string name = tokens[n].substr(name_string.length());
    const std::string opening_quotes = "\"'{";
    if (!name.empty() && opening_quotes.find(name.front()) != std::string::npos) {
      const char closing_quote = (name.front() == '{') ? '}' : name.front();
      // the comment line is split at spaces, so a quoted value with spaces spans several tokens
      if (name.size() < 2 || name.back() != closing_quote) {
        for (int m = n + 1; m < tokens.size() && (name.size() < 2 || name.back() != closing_quote);
             ++m) {
          name += " " + tokens[m];
        }
        const bool is_closed = name.size() >= 2 && name.back() == closing_quote;
        const std::string message =
          is_closed ? location + ": the name " + name +
                        " contains whitespace, to which ediff.in cannot refer."
                    : location + ": the value of name= opens a quote that the line does not close.";
        PRINT_INPUT_ERROR(message.c_str());
      }
      name = name.substr(1, name.size() - 2);
    }
    if (!is_valid_structure_name(name)) {
      const std::string message = location + ": the name " + name +
                                  " cannot be referred to in ediff.in. A name must not be empty, "
                                  "begin with #, + or -, or contain * / = \" ' { or }.";
      PRINT_INPUT_ERROR(message.c_str());
    }
    structure_name = name;
  }
  return structure_name;
}

void EnergyDifference::read_train(
  Parameters& para,
  std::vector<Structure>& structures_train,
  const int batch_size,
  int& num_batches,
  std::vector<int>& batch_sizes)
{
  if (para.prediction == 0) {
    std::ifstream ediff_file("ediff.in");
    if (para.lambda_d > 0.0f) {
      if (!ediff_file.is_open()) {
        PRINT_INPUT_ERROR("lambda_d requires the file ediff.in.");
      }
      entries = read_ediff_in(ediff_file);
      is_entry_in_train.assign(entries.size(), false);
      is_entry_in_test.assign(entries.size(), false);
      // read_structures leaves train.xyz in file order for this grouping
      if (num_batches > 1) {
        batch_sizes =
          group_structures_by_combination(entries, structures_train, batch_size, num_batches);
      }
      combinations_train = resolve_ediff_combinations(
        entries, structures_train, para, batch_sizes, "train.xyz", is_entry_in_train);
      if (combinations_train.empty()) {
        PRINT_INPUT_ERROR("No combination in ediff.in has all its structures in train.xyz.");
      }
      para.has_ediff_combinations = true;
    } else if (ediff_file.is_open()) {
      printf("ediff.in is ignored because lambda_d is not set.\n");
    }
  }
}

void EnergyDifference::read_test(
  const Parameters& para, const std::vector<Structure>& structures_test, const bool has_test_set)
{
  if (!para.has_ediff_combinations) {
    return;
  }
  if (has_test_set) {
    combinations_test = resolve_ediff_combinations(
      entries, structures_test, para, {(int)structures_test.size()}, "test.xyz", is_entry_in_test);
  }
  int num_in_both = 0;
  int num_skipped = 0;
  int first_skipped = -1;
  for (int k = 0; k < (int)entries.size(); ++k) {
    if (is_entry_in_train[k] && is_entry_in_test[k]) {
      ++num_in_both;
    } else if (!is_entry_in_train[k] && !is_entry_in_test[k]) {
      if (num_skipped++ == 0) {
        first_skipped = k;
      }
    }
  }
  printf(
    "ediff.in: %d combinations, %d in train.xyz, %d in test.xyz (%d in both), %d skipped.\n",
    (int)entries.size(),
    (int)combinations_train.size(),
    (int)combinations_test.size(),
    num_in_both,
    num_skipped);
  if (num_skipped > 0) {
    printf(
      "Warning: %d combination(s) of ediff.in skipped, whose structures are neither all in "
      "train.xyz nor all in test.xyz, e.g. line %d.\n",
      num_skipped,
      entries[first_skipped].line_number);
  }
  if (has_test_set && combinations_test.empty()) {
    printf("Warning: no combination of ediff.in lies in test.xyz, so rmse_ediff_test is 0.\n");
  }
}

float EnergyDifference::get_rmse_train(
  Dataset& dataset, const int batch_id, const int device_id) const
{
  return get_rmse(combinations_train, dataset, batch_id, device_id);
}

float EnergyDifference::get_rmse_test(Dataset& dataset, const int device_id) const
{
  return get_rmse(combinations_test, dataset, 0, device_id);
}
