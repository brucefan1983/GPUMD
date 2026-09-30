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
EnergyDifference::Entry
EnergyDifference::parse_line(const std::vector<std::string>& tokens, const int line_number)
{
  Entry entry;
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
    Term energy_diff_term;
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

std::vector<EnergyDifference::Entry> EnergyDifference::read_entries(std::ifstream& input)
{
  std::vector<Entry> entries;
  int line_number = 0;
  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens_without_comments(input);
    ++line_number;
    if (!tokens.empty()) {
      entries.push_back(parse_line(tokens, line_number));
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

// Returns the groups of structures that linked_indices, the structures of each combination, link
// directly or through other combinations, each group in the order of the structures.
static std::vector<std::vector<int>>
find_groups(const std::vector<std::vector<int>>& linked_indices, const int n_total)
{
  std::vector<int> parent(n_total);
  std::iota(parent.begin(), parent.end(), 0);
  auto find_root = [&parent](int n) {
    while (parent[n] != n) {
      parent[n] = parent[parent[n]];
      n = parent[n];
    }
    return n;
  };
  for (const auto& indices : linked_indices) {
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
  return groups;
}

/*----------------------------------------------------------------------------80
Reorders the training structures into num_batches batches that keep each group together, and
returns the batch sizes. The groups, the larger ones first and those of one size by their mean
energy per atom, go one by one to the batch with the fewest structures, which for groups of one
structure gives the batches of read_structures.
Batches left empty are dropped, so num_batches can decrease.
------------------------------------------------------------------------------*/
static std::vector<int> place_groups_in_batches(
  const std::vector<std::vector<int>>& groups,
  std::vector<Structure>& structures,
  const int batch_size,
  int& num_batches)
{
  const int n_total = structures.size();
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

// Stops with an input error when two structures of a combination, given by their names and
// structures, are the same structure for the model.
static void check_distinct_structures(
  const std::vector<std::string>& names,
  const std::vector<const Structure*>& members,
  const Parameters& para,
  const int line_number,
  const char* xyz_filename)
{
  for (int i = 0; i < (int)members.size(); ++i) {
    for (int j = i + 1; j < (int)members.size(); ++j) {
      if (!are_the_same_structure(*members[i], *members[j], para.model_type)) {
        continue;
      }
      const std::string pair = names[i] + " and " + names[j] + " in " + xyz_filename;
      const bool is_charge_model = para.charge_mode || para.charge_vdw;
      if (is_charge_model && members[i]->charge != members[j]->charge) {
        print_ediff_in_error(
          line_number,
          pair + " differ only in charge=, for which a qNEP model predicts no meaningful energy "
                 "difference.");
      }
      print_ediff_in_error(
        line_number,
        pair + " have the same geometry, for which the model predicts the same energy.");
    }
  }
}

// Stops with an input error when a combination leaves more than 1e-6 atoms of a type over, since
// the atoms left over carry the free energy offset per atom of the population.
static void check_balance(
  const std::vector<double>& coefficients,
  const std::vector<const Structure*>& members,
  const std::vector<std::string>& elements,
  const int line_number,
  const char* xyz_filename)
{
  std::vector<double> imbalance(elements.size(), 0.0);
  for (int i = 0; i < (int)members.size(); ++i) {
    for (const int type : members[i]->type) {
      imbalance[type] += coefficients[i];
    }
  }
  for (int t = 0; t < (int)elements.size(); ++t) {
    if (std::fabs(imbalance[t]) <= 1.0e-6) {
      continue;
    }
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
    print_ediff_in_error(line_number, message);
  }
}

// Returns the combinations of the entries whose names all label structures of one data set,
// split into batches of the given sizes, and marks those entries in is_in_set. The reference of a
// combination is the same combination of the reference total energies.
std::vector<EnergyDifference::Combination> EnergyDifference::resolve(
  const std::vector<Entry>& entries,
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

  std::vector<Combination> combinations;
  for (int k = 0; k < (int)entries.size(); ++k) {
    const Entry& entry = entries[k];
    const bool is_resolved =
      std::all_of(entry.terms.begin(), entry.terms.end(), [&](const Term& term) {
        return name_to_index.count(term.name) > 0;
      });
    if (!is_resolved) {
      continue;
    }
    std::vector<std::string> names;
    std::vector<const Structure*> members;
    Combination combination;
    combination.batch = index_to_batch[name_to_index[entry.terms[0].name]];
    combination.ref_total_eV = 0.0;
    combination.weight = entry.weight;
    for (const auto& term : entry.terms) {
      const int index = name_to_index[term.name];
      names.push_back(term.name);
      members.push_back(&structures[index]);
      combination.local.push_back(index_to_local[index]);
      combination.coefficient.push_back(term.coefficient);
      combination.ref_total_eV += term.coefficient * structures[index].energy_total;
    }
    check_distinct_structures(names, members, para, entry.line_number, xyz_filename);
    check_balance(combination.coefficient, members, para.elements, entry.line_number, xyz_filename);
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
float EnergyDifference::get_rmse(
  const std::vector<Combination>& combinations,
  Dataset& dataset,
  const int batch_id,
  const int device_id)
{
  const bool is_any_in_batch = std::any_of(
    combinations.begin(), combinations.end(), [batch_id](const Combination& combination) {
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

/*----------------------------------------------------------------------------80
Scans the key=value fields of an extended XYZ comment line. A value that begins with a double
quote, a single quote, a brace or a bracket extends to the matching closing character, and a
backslash escapes the character after it.
------------------------------------------------------------------------------*/
std::string EnergyDifference::read_structure_name(
  const std::string& comment_line, const std::string& xyz_filename, const int line_number)
{
  const std::string location = xyz_filename + " line " + std::to_string(line_number);
  const std::string opening_delimiters = "\"'{[";
  const std::string closing_delimiters = "\"'}]";
  const size_t length = comment_line.size();
  auto is_space = [&comment_line](size_t n) {
    return std::isspace(static_cast<unsigned char>(comment_line[n])) != 0;
  };
  std::string structure_name;
  bool has_name = false;
  size_t n = 0;
  while (true) {
    while (n < length && is_space(n)) {
      ++n;
    }
    if (n == length) {
      break;
    }
    const size_t key_start = n;
    while (n < length && comment_line[n] != '=' && !is_space(n)) {
      ++n;
    }
    const std::string key = to_lowercase(comment_line.substr(key_start, n - key_start));
    while (n < length && is_space(n)) {
      ++n;
    }
    if (n == length || comment_line[n] != '=') {
      continue;
    }
    ++n;
    while (n < length && is_space(n)) {
      ++n;
    }
    const size_t value_start = n;
    const size_t kind = n < length ? opening_delimiters.find(comment_line[n]) : std::string::npos;
    bool is_closed = true;
    if (kind != std::string::npos) {
      is_closed = false;
      for (++n; n < length && !is_closed; ++n) {
        if (comment_line[n] == '\\') {
          ++n;
        } else if (comment_line[n] == closing_delimiters[kind]) {
          is_closed = true;
        }
      }
    } else {
      while (n < length && !is_space(n)) {
        ++n;
      }
    }
    if (key != "name") {
      continue;
    }
    if (has_name) {
      PRINT_INPUT_ERROR((location + ": more than one name= field.").c_str());
    }
    if (!is_closed) {
      PRINT_INPUT_ERROR(
        (location + ": the value of name= opens a quote that the line does not close.").c_str());
    }
    const std::string value = to_lowercase(comment_line.substr(value_start, n - value_start));
    const std::string name =
      (kind != std::string::npos) ? value.substr(1, value.size() - 2) : value;
    if (name.find_first_of(" \t") != std::string::npos) {
      const std::string message =
        location + ": the name " + value + " contains whitespace, to which ediff.in cannot refer.";
      PRINT_INPUT_ERROR(message.c_str());
    }
    if (!is_valid_structure_name(name)) {
      const std::string message = location + ": the name " + name +
                                  " cannot be referred to in ediff.in. A name must not be empty, "
                                  "begin with #, + or -, or contain * / = \" ' { or }.";
      PRINT_INPUT_ERROR(message.c_str());
    }
    structure_name = name;
    has_name = true;
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
      entries = read_entries(ediff_file);
      is_entry_in_train.assign(entries.size(), false);
      is_entry_in_test.assign(entries.size(), false);
      // read_structures leaves train.xyz in file order for this grouping
      if (num_batches > 1) {
        auto name_to_index = get_name_to_index(structures_train, "train.xyz");
        std::vector<std::vector<int>> linked_indices;
        for (const auto& entry : entries) {
          std::vector<int> indices;
          for (const auto& term : entry.terms) {
            if (name_to_index.count(term.name) > 0) {
              indices.push_back(name_to_index[term.name]);
            }
          }
          if (indices.size() == entry.terms.size()) {
            linked_indices.push_back(indices);
          }
        }
        const auto groups = find_groups(linked_indices, structures_train.size());
        batch_sizes = place_groups_in_batches(groups, structures_train, batch_size, num_batches);
      }
      combinations_train =
        resolve(entries, structures_train, para, batch_sizes, "train.xyz", is_entry_in_train);
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
    combinations_test = resolve(
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
