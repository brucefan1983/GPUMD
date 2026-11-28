/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Read fixed-topology molecular parameters and atom assignments.

Major contributors:
  Zheyong Fan: Initial implementation.
------------------------------------------------------------------------------*/

#include "input.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <utility>

namespace
{

class Records
{
public:
  explicit Records(const char* filename) : filename(filename), input(filename)
  {
    if (!input) {
      fail("cannot open file");
    }
  }

  void fail(const std::string& reason) const
  {
    PRINT_INPUT_ERROR((filename + ":" + std::to_string(line_number) + ": " + reason).c_str());
  }

  bool next()
  {
    std::string line;
    while (std::getline(input, line)) {
      ++line_number;
      fields = get_tokens_without_comments(line);
      if (!fields.empty()) {
        return true;
      }
    }
    ++line_number;
    if (input.bad()) {
      fail("error reading file");
    }
    return false;
  }

  void row(size_t count, const char* format)
  {
    if (!next() || fields.size() != count) {
      fail(std::string("expected ") + format);
    }
  }

  int count(const char* keyword)
  {
    row(2, keyword);
    if (fields[0] != keyword) {
      fail(std::string("expected section ") + keyword);
    }
    return integer(1);
  }

  int integer(size_t column) const
  {
    // Interpret numbered records as decimal, including entries with leading zeros.
    const std::string& token = fields[column];
    const size_t first = token.find_first_not_of('0');
    int value = 0;
    if (
      token.find_first_not_of("0123456789") != std::string::npos ||
      (first != std::string::npos && !is_valid_int(token.c_str() + first, &value))) {
      fail("expected a nonnegative integer, found '" + token + "'");
    }
    return value;
  }

  double real(size_t column) const
  {
    double value;
    if (!is_valid_real(fields[column], &value)) {
      fail("expected a finite number, found '" + fields[column] + "'");
    }
    return value;
  }

  int section_count(std::set<std::string>& seen) const
  {
    if (fields.size() != 2) {
      fail("expected a section name and count");
    }
    if (!seen.insert(fields[0]).second) {
      fail("unexpected repeated section '" + fields[0] + "'");
    }
    return integer(1);
  }

  std::vector<std::string> fields;

private:
  std::string filename;
  std::ifstream input;
  size_t line_number = 0;
};

void read_harmonic_types(Records& file, int count, bool is_angle, MolecularInput& data)
{
  const char* format = is_angle ? "<angle type ID> <force constant> <equilibrium angle in degrees>"
                                : "<bond type ID> <force constant> <equilibrium length>";
  std::map<int, std::pair<double, double>> parameters;
  for (int t = 0; t < count; ++t) {
    file.row(3, format);
    const int id = file.integer(0);
    const double k = file.real(1);
    double equilibrium = file.real(2);
    if (k < 0.0) {
      file.fail("harmonic force constant must be nonnegative");
    }
    if (is_angle) {
      if (equilibrium < 0.0 || equilibrium > 180.0) {
        file.fail("equilibrium angle must lie in [0, 180] degrees");
      }
      equilibrium *= std::acos(-1.0) / 180.0;
    } else if (equilibrium <= 0.0) {
      file.fail("bond equilibrium length must be positive");
    }
    if (id >= count || !parameters.emplace(id, std::make_pair(k, equilibrium)).second) {
      file.fail(
        is_angle ? "invalid or duplicate angle type ID" : "invalid or duplicate bond type ID");
    }
  }
  auto& constants = is_angle ? data.angle_k : data.bond_k;
  auto& equilibria = is_angle ? data.angle_theta0 : data.bond_r0;
  for (const auto& parameter : parameters) {
    constants.push_back(parameter.second.first);
    equilibria.push_back(parameter.second.second);
  }
}

std::vector<std::string> read_forcefield(const char* filename, MolecularInput& data)
{
  Records file(filename);
  if (
    !file.next() || file.fields.size() < 3 || file.fields[0] != "molecular" ||
    static_cast<size_t>(file.integer(1)) != file.fields.size() - 2) {
    file.fail("expected molecular <element count> <element symbols>");
  }
  const std::set<std::string> elements(file.fields.begin() + 2, file.fields.end());
  if (elements.size() != file.fields.size() - 2) {
    file.fail("duplicate element in molecular header");
  }

  file.row(2, "cutoff <distance>");
  if (
    file.fields[0] != "cutoff" || (data.cutoff = file.real(1)) <= 0.0 ||
    !std::isfinite(data.cutoff * data.cutoff) || data.cutoff * data.cutoff == 0.0) {
    file.fail("cutoff must be positive with a representable squared distance");
  }
  data.num_types = file.count("types");
  if (data.num_types == 0 || data.num_types > std::numeric_limits<int>::max() / data.num_types) {
    file.fail("the LJ type table must fit in the kernel index range");
  }
  std::vector<std::string> symbols(data.num_types);
  for (int t = 0; t < data.num_types; ++t) {
    file.row(2, "<type ID> <element>");
    const int id = file.integer(0);
    if (id >= data.num_types || !symbols[id].empty() || !elements.count(file.fields[1])) {
      file.fail("invalid or duplicate type ID, or element missing from molecular header");
    }
    symbols[id] = file.fields[1];
  }

  const int count = file.count("pairs");
  if (count != static_cast<long long>(data.num_types) * (data.num_types + 1) / 2) {
    file.fail("pairs must contain every upper-triangular type pair exactly once");
  }
  const int table_size = data.num_types * data.num_types;
  data.c6.assign(table_size, -1.0);
  data.c12.resize(table_size);
  for (int pair = 0; pair < count; ++pair) {
    file.row(4, "<type i> <type j> <sigma> <epsilon>");
    const int i = file.integer(0);
    const int j = file.integer(1);
    if (i > j || j >= data.num_types) {
      file.fail("type pair must satisfy 0 <= i <= j < type count");
    }
    const int ij = i * data.num_types + j;
    const int ji = j * data.num_types + i;
    if (data.c6[ij] >= 0.0) {
      file.fail("duplicate LJ type pair");
    }
    const double sigma = file.real(2);
    const double epsilon = file.real(3);
    if (sigma < 0.0 || epsilon < 0.0 || (epsilon > 0.0 && sigma == 0.0)) {
      file.fail("sigma and epsilon must be nonnegative; positive epsilon requires positive sigma");
    }
    double c6 = 0.0;
    double c12 = 0.0;
    if (epsilon > 0.0) {
      const double s2 = sigma * sigma;
      const double s6 = s2 * s2 * s2;
      c6 = 4.0 * epsilon * s6;
      c12 = c6 * s6;
      if (!std::isfinite(c6) || !std::isfinite(c12) || c6 == 0.0 || c12 == 0.0) {
        file.fail("LJ coefficients are outside the representable range");
      }
    }
    data.c6[ij] = data.c6[ji] = c6;
    data.c12[ij] = data.c12[ji] = c12;
  }
  std::set<std::string> seen;
  while (file.next()) {
    const int entries = file.section_count(seen);
    if (file.fields[0] != "angle_types" && file.fields[0] != "bond_types") {
      file.fail("unexpected force-field section '" + file.fields[0] + "'");
    }
    read_harmonic_types(file, entries, file.fields[0] == "angle_types", data);
  }
  return symbols;
}

template <typename T>
void store_pairs(
  int n,
  const std::map<std::pair<int, int>, T>& pairs,
  std::vector<int>& offsets,
  std::vector<int>& partners,
  std::vector<T>& values)
{
  offsets.resize(static_cast<size_t>(n) + 1, 0);
  for (const auto& pair : pairs) {
    ++offsets[pair.first.first + 1];
    ++offsets[pair.first.second + 1];
  }
  std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
  partners.resize(2 * pairs.size());
  values.resize(partners.size());
  std::vector<int> next = offsets;
  // Ordered (i,j), i<j: each row receives its smaller partners first, then its larger partners.
  for (const auto& pair : pairs) {
    const int i = pair.first.first;
    const int j = pair.first.second;
    const int ij = next[i]++;
    const int ji = next[j]++;
    partners[ij] = j;
    partners[ji] = i;
    values[ij] = values[ji] = pair.second;
  }
}

std::pair<int, int> atom_pair(Records& file, int n)
{
  const int i = file.integer(0);
  const int j = file.integer(1);
  if (i >= n || j >= n || i == j) {
    file.fail("expected two distinct atom IDs from model.xyz");
  }
  return std::minmax(i, j);
}

void read_special_pairs(Records& file, int n, int count, MolecularInput& data)
{
  std::map<std::pair<int, int>, double> pairs;
  for (int p = 0; p < count; ++p) {
    file.row(4, "<atom i> <atom j> <LJ weight> <Coulomb weight>");
    const auto pair = atom_pair(file, n);
    const double lj = file.real(2);
    const double coulomb = file.real(3);
    if (lj < 0.0 || lj > 1.0 || coulomb < 0.0 || coulomb > 1.0) {
      file.fail("special pair weights must lie in [0, 1]");
    }
    if (!pairs.emplace(pair, lj).second) {
      file.fail("duplicate special pair");
    }
  }
  store_pairs(n, pairs, data.offsets, data.partners, data.lj_weights);
}

void read_bonds(Records& file, int n, int count, MolecularInput& data)
{
  if (count == 0) {
    return;
  }
  if (data.bond_k.empty()) {
    file.fail("bonds require bond_types in the force-field file");
  }
  std::map<std::pair<int, int>, int> pairs;
  for (int p = 0; p < count; ++p) {
    file.row(3, "<atom i> <atom j> <bond type ID>");
    const auto pair = atom_pair(file, n);
    const int type = file.integer(2);
    if (static_cast<size_t>(type) >= data.bond_k.size()) {
      file.fail("bond type ID is out of range");
    }
    if (!pairs.emplace(pair, type).second) {
      file.fail("duplicate bond");
    }
  }
  store_pairs(n, pairs, data.bond_offsets, data.bond_partners, data.bond_types);
}

void read_angles(Records& file, int n, int count, MolecularInput& data)
{
  if (count == 0) {
    return;
  }
  const long long pairs_per_center = static_cast<long long>(n - 1) * (n - 2) / 2;
  // Compare ceil(count / n) to avoid overflowing the number of possible triplets.
  if (
    count > std::numeric_limits<int>::max() / 3 ||
    (static_cast<long long>(count) + n - 1) / n > pairs_per_center) {
    file.fail("angle count exceeds the number of possible or indexable angles");
  }
  if (data.angle_k.empty()) {
    file.fail("angles require angle_types in the force-field file");
  }
  std::set<std::array<int, 3>> seen;
  data.angle_offsets.resize(static_cast<size_t>(n) + 1, 0);
  for (int a = 0; a < count; ++a) {
    file.row(4, "<atom i> <central atom j> <atom k> <angle type ID>");
    const MolecularAngle angle = {
      file.integer(0), file.integer(1), file.integer(2), file.integer(3)};
    if (
      angle.first >= n || angle.center >= n || angle.last >= n || angle.first == angle.center ||
      angle.last == angle.center || angle.first == angle.last) {
      file.fail("angle requires three distinct atom IDs from model.xyz");
    }
    if (static_cast<size_t>(angle.type) >= data.angle_k.size()) {
      file.fail("angle type ID is out of range");
    }
    const std::array<int, 3> key = {
      std::min(angle.first, angle.last), angle.center, std::max(angle.first, angle.last)};
    if (!seen.insert(key).second) {
      file.fail("duplicate angle");
    }
    data.angles.push_back(angle);
    ++data.angle_offsets[angle.first + 1];
    ++data.angle_offsets[angle.center + 1];
    ++data.angle_offsets[angle.last + 1];
  }
  std::partial_sum(
    data.angle_offsets.begin(), data.angle_offsets.end(), data.angle_offsets.begin());
  data.angle_indices.resize(3 * count);
  std::vector<int> next = data.angle_offsets;
  for (int a = 0; a < count; ++a) {
    const MolecularAngle& angle = data.angles[a];
    data.angle_indices[next[angle.first]++] = a;
    data.angle_indices[next[angle.center]++] = a;
    data.angle_indices[next[angle.last]++] = a;
  }
}

void read_topology_sections(Records& file, int n, MolecularInput& data)
{
  std::set<std::string> seen;
  while (file.next()) {
    const int count = file.section_count(seen);
    if (file.fields[0] == "angles") {
      read_angles(file, n, count, data);
    } else if (file.fields[0] == "bonds" || file.fields[0] == "special_pairs") {
      if (
        count > std::numeric_limits<int>::max() / 2 ||
        count > static_cast<long long>(n) * (n - 1) / 2) {
        file.fail("pair count exceeds the number of possible or indexable pairs");
      }
      if (file.fields[0] == "bonds") {
        read_bonds(file, n, count, data);
      } else {
        read_special_pairs(file, n, count, data);
      }
    } else {
      file.fail("unexpected topology section '" + file.fields[0] + "'");
    }
  }
}

} // namespace

MolecularInput read_molecular_input(
  const char* forcefield,
  const char* topology,
  const int number_of_atoms,
  const std::vector<std::string>& atom_symbols)
{
  MolecularInput data;
  const std::vector<std::string> symbols = read_forcefield(forcefield, data);
  Records file(topology);
  const int n = file.count("atoms");
  if (n == 0 || n != number_of_atoms || static_cast<size_t>(n) != atom_symbols.size()) {
    file.fail("atom count must match model.xyz");
  }
  data.types.resize(n);
  for (int i = 0; i < n; ++i) {
    file.row(2, "<force-field type ID> <charge>");
    const int type = file.integer(0);
    if (type >= data.num_types || symbols[type] != atom_symbols[i]) {
      file.fail("force-field type does not match atom " + std::to_string(i) + " in model.xyz");
    }
    if (file.real(1) != 0.0) {
      file.fail("molecular LJ requires zero atomic charges");
    }
    data.types[i] = type;
  }
  read_topology_sections(file, n, data);
  return data;
}
