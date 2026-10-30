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

#include "dataset.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
void draw_batch(
  std::vector<int>& permutation,
  std::size_t& cursor,
  const int batch_size,
  std::mt19937& rng,
  std::vector<int>& batch)
{
  const std::size_t n = permutation.size();
  if (n == 0 || batch_size <= 0 || static_cast<std::size_t>(batch_size) > n) {
    PRINT_INPUT_ERROR("Invalid cursor batch request.");
  }

  batch.clear();
  batch.reserve(batch_size);

  if (cursor + batch_size <= n) {
    for (int t = 0; t < batch_size; ++t) {
      batch.push_back(permutation[cursor + t]);
    }
    cursor += batch_size;
    if (cursor == n) {
      cursor = 0;
      std::shuffle(permutation.begin(), permutation.end(), rng);
    }
    return;
  }

  std::vector<int> old_tail;
  for (std::size_t p = cursor; p < n; ++p) {
    batch.push_back(permutation[p]);
    old_tail.push_back(permutation[p]);
  }

  std::shuffle(permutation.begin(), permutation.end(), rng);
  std::stable_partition(permutation.begin(), permutation.end(), [&](int value) {
    return std::find(old_tail.begin(), old_tail.end(), value) == old_tail.end();
  });

  const int need = batch_size - static_cast<int>(batch.size());
  for (int t = 0; t < need; ++t) {
    batch.push_back(permutation[t]);
  }
  cursor = need;
}

class XYZInput
{
public:
  explicit XYZInput(const std::string& filename, bool optional) : filename_(filename)
  {
    errno = 0;
    input_.open(filename);
    if (!input_.is_open() && !(optional && errno == ENOENT)) {
      fail("Failed to open dataset");
    }
  }

  bool is_open() const { return input_.is_open(); }

  bool read(Dataset& data, const Parameters& para)
  {
    std::string line;
    std::vector<std::string> tokens;
    while (std::getline(input_, line)) {
      ++line_;
      tokens = get_tokens(line);
      if (!tokens.empty()) {
        break;
      }
    }
    if (tokens.empty()) {
      return false;
    }
    if (tokens.size() != 1) {
      fail("Expected one atom count");
    }
    const int na = positive_int(tokens[0]);
    const auto fields = header(read_line());
    const int nk = positive_int(field(fields, "nkpoints"));
    const bool has_bands = fields.count("bands") != 0;
    if (has_bands != (fields.count("nbands") != 0)) {
      fail("bands and nbands must be supplied together");
    }
    if (!has_bands && !para.prediction) {
      fail("Training and test data require reference bands and nbands");
    }
    const int nb = has_bands ? positive_int(field(fields, "nbands")) : 0;
    if (
      para.band_min < 0 || para.band_max <= para.band_min || (has_bands && para.band_max > nb) ||
      para.band_max - para.band_min > static_cast<long long>(na) * para.orbitals_per_atom) {
      fail("band_range exceeds reference or TB band capacity");
    }
    auto values = get_tokens(field(fields, "lattice"));
    if (values.size() != 9) {
      fail("Lattice requires nine values");
    }
    for (int i = 0; i < 9; ++i) {
      data.structure.lattice[i / 3][i % 3] = real(values[i]);
    }
    values = get_tokens(field(fields, "kpoints"));
    if (values.size() != static_cast<std::size_t>(nk) * 3) {
      fail("kpoints requires 3 * nkpoints values");
    }
    data.kpoints.resize(nk);
    for (int k = 0; k < nk; ++k) {
      for (int d = 0; d < 3; ++d) {
        data.kpoints[k][d] = real(values[3 * static_cast<std::size_t>(k) + d]);
      }
    }
    if (has_bands) {
      values = get_tokens(field(fields, "bands"));
      if (values.size() != static_cast<std::size_t>(nk) * nb) {
        fail("bands requires nkpoints * nbands values");
      }
      data.bands.assign(nk, std::vector<double>(para.band_max - para.band_min));
      for (int k = 0; k < nk; ++k) {
        for (int n = 0; n < nb; ++n) {
          const double value = real(values[static_cast<std::size_t>(k) * nb + n]);
          if (n >= para.band_min && n < para.band_max) {
            data.bands[k][n - para.band_min] = value;
          }
        }
      }
    }
    const auto pbc = fields.find("pbc");
    if (pbc != fields.end()) {
      values = get_tokens(pbc->second);
      if (values.size() != 3) {
        fail("pbc requires three true values");
      }
      for (auto value : values) {
        lower(value);
        if (value != "t" && value != "true" && value != "1") {
          fail("TB requires periodicity in all three lattice directions");
        }
      }
    }
    std::string properties = field(fields, "properties");
    std::replace(properties.begin(), properties.end(), ':', ' ');
    values = get_tokens(properties);
    if (values.size() % 3 != 0) {
      fail("Invalid Properties declaration");
    }
    int columns = 0, species = -1, position = -1;
    for (std::size_t i = 0; i < values.size(); i += 3) {
      const int count = positive_int(values[i + 2]);
      if (count > std::numeric_limits<int>::max() - columns) {
        fail("Too many Properties columns");
      }
      if (values[i] == "species") {
        if (species >= 0 || values[i + 1] != "S" || count != 1) {
          fail("Expected species:S:1 once");
        }
        species = columns;
      } else if (values[i] == "pos") {
        if (position >= 0 || values[i + 1] != "R" || count != 3) {
          fail("Expected pos:R:3 once");
        }
        position = columns;
      }
      columns += count;
    }
    if (species < 0 || position < 0) {
      fail("Properties must include species:S:1 and pos:R:3");
    }
    data.structure.num_atoms = na;
    for (int i = 0; i < na; ++i) {
      tokens = get_tokens(read_line());
      if (tokens.size() != static_cast<std::size_t>(columns)) {
        fail("Atom row does not match Properties");
      }
      Dataset::Atom atom;
      atom.symbol = tokens[species];
      const auto type = std::find(para.elements.begin(), para.elements.end(), atom.symbol);
      if (type == para.elements.end()) {
        fail("Unknown element '" + atom.symbol + "'; it is not listed by type");
      }
      atom.type = static_cast<int>(type - para.elements.begin());
      for (int d = 0; d < 3; ++d) {
        atom.position[d] = real(tokens[position + d]);
      }
      data.structure.atoms.push_back(atom);
    }
    return true;
  }

private:
  void fail(const std::string& message) const
  {
    PRINT_INPUT_ERROR((message + " at " + filename_ + ":" + std::to_string(line_) + ".").c_str());
  }

  std::string read_line()
  {
    std::string line;
    ++line_;
    if (!std::getline(input_, line)) {
      fail("Unexpected end of XYZ frame");
    }
    return line;
  }

  int positive_int(const std::string& token) const
  {
    std::istringstream input(token);
    int value = 0;
    if (!(input >> value) || input.peek() != std::char_traits<char>::eof() || value <= 0) {
      fail("Expected positive integer, got '" + token + "'");
    }
    return value;
  }

  double real(const std::string& token) const
  {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(token.c_str(), &end);
    if (end == token.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(value)) {
      fail("Invalid number '" + token + "'");
    }
    return value;
  }

  static void lower(std::string& text)
  {
    for (char& c : text) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
  }

  std::map<std::string, std::string> header(const std::string& line) const
  {
    std::istringstream input(line);
    std::map<std::string, std::string> fields;
    std::string key, value;
    while (std::getline(input >> std::ws, key, '=')) {
      const auto tokens = get_tokens(key);
      if (tokens.size() != 1 || input.eof()) {
        fail("Expected key=value in XYZ header");
      }
      key = tokens[0];
      input >> std::ws;
      if (input.peek() == '\"' || input.peek() == '\'') {
        const char quote = static_cast<char>(input.get());
        if (
          !std::getline(input, value, quote) || input.eof() ||
          (input.peek() != std::char_traits<char>::eof() &&
           !std::isspace(static_cast<unsigned char>(input.peek())))) {
          fail("Unclosed or misplaced quote in XYZ header");
        }
      } else if (!(input >> value)) {
        fail("Missing XYZ header value");
      }
      lower(key);
      if (!fields.emplace(key, value).second) {
        fail("Duplicate XYZ header key '" + key + "'");
      }
    }
    return fields;
  }

  const std::string&
  field(const std::map<std::string, std::string>& fields, const std::string& key) const
  {
    const auto entry = fields.find(key);
    if (entry == fields.end()) {
      fail("Missing XYZ header key '" + key + "'");
    }
    return entry->second;
  }

  std::string filename_;
  std::ifstream input_;
  int line_ = 0;
};

double determinant(const std::array<std::array<double, 3>, 3>& a)
{
  return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
         a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
         a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
}

std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b)
{
  return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
}

double norm(const std::array<double, 3>& a)
{
  return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

bool canonical_bond(int i, int j, int sx, int sy, int sz)
{
  if (sx > 0) {
    return true;
  }
  if (sx < 0) {
    return false;
  }
  if (sy > 0) {
    return true;
  }
  if (sy < 0) {
    return false;
  }
  if (sz > 0) {
    return true;
  }
  if (sz < 0) {
    return false;
  }
  return j > i;
}
} // namespace

DatasetCollection::DatasetCollection(
  const Parameters& para, const std::string& filename, bool optional)
{
  XYZInput input(filename, optional);
  if (!input.is_open()) {
    return;
  }
  while (true) {
    Dataset data;
    if (!input.read(data, para)) {
      break;
    }
    if (!structures_.empty() && has_reference() != !data.bands.empty()) {
      PRINT_INPUT_ERROR(
        (filename + ": frames must consistently include or omit reference bands.").c_str());
    }
    data.index = static_cast<int>(structures_.size());
    data.structure.volume = std::abs(determinant(data.structure.lattice));
    if (!std::isfinite(data.structure.volume) || data.structure.volume < 1.0e-12) {
      PRINT_INPUT_ERROR(
        ("Invalid lattice in " + filename + ", frame " + std::to_string(data.index + 1) + ".")
          .c_str());
    }
    data.build_cartesian_kpoints();
    data.build_neighbors(para);
    structures_.push_back(std::move(data));
  }
  if (structures_.empty()) {
    PRINT_INPUT_ERROR(("No XYZ frames in " + filename + ".").c_str());
  }
}

void Dataset::build_cartesian_kpoints()
{
  constexpr double two_pi = 6.2831853071795864769;
  const auto& a1 = structure.lattice[0];
  const auto& a2 = structure.lattice[1];
  const auto& a3 = structure.lattice[2];
  const auto c23 = cross(a2, a3);
  const auto c31 = cross(a3, a1);
  const auto c12 = cross(a1, a2);
  const double signed_volume = a1[0] * c23[0] + a1[1] * c23[1] + a1[2] * c23[2];

  std::array<std::array<double, 3>, 3> b{};
  for (int d = 0; d < 3; ++d) {
    b[0][d] = two_pi * c23[d] / signed_volume;
    b[1][d] = two_pi * c31[d] / signed_volume;
    b[2][d] = two_pi * c12[d] / signed_volume;
  }

  kpoints_cart.resize(kpoints.size());
  for (std::size_t ik = 0; ik < kpoints.size(); ++ik) {
    for (int d = 0; d < 3; ++d) {
      kpoints_cart[ik][d] =
        kpoints[ik][0] * b[0][d] + kpoints[ik][1] * b[1][d] + kpoints[ik][2] * b[2][d];
    }
  }
}

void Dataset::build_neighbors(const Parameters& para)
{
  neighbors_radial.clear();
  neighbors_angular.clear();
  hopping_bonds.clear();

  const auto& a = structure.lattice[0];
  const auto& b = structure.lattice[1];
  const auto& c = structure.lattice[2];

  const double h_a = structure.volume / norm(cross(b, c));
  const double h_b = structure.volume / norm(cross(c, a));
  const double h_c = structure.volume / norm(cross(a, b));
  const double rc_max = std::max(para.rc_radial, std::max(para.rc_angular, para.rc_hopping));

  auto image_count = [&](double height) {
    const double count = std::ceil(rc_max / height) + 1.0;
    if (!(height > 0.0) || !std::isfinite(count) || count >= std::numeric_limits<int>::max()) {
      PRINT_INPUT_ERROR("Cell and cutoff require an invalid number of periodic images.");
    }
    return static_cast<int>(count);
  };
  const int na = image_count(h_a);
  const int nb = image_count(h_b);
  const int nc = image_count(h_c);

  const double rc_radial2 = para.rc_radial * para.rc_radial;
  const double rc_angular2 = para.rc_angular * para.rc_angular;
  const double rc_hopping2 = para.rc_hopping * para.rc_hopping;
  const double tiny = 1.0e-20;

  for (int i = 0; i < structure.num_atoms; ++i) {
    const auto& ri = structure.atoms[i].position;
    for (int j = 0; j < structure.num_atoms; ++j) {
      const auto& rj = structure.atoms[j].position;
      for (int sx = -na; sx <= na; ++sx) {
        for (int sy = -nb; sy <= nb; ++sy) {
          for (int sz = -nc; sz <= nc; ++sz) {
            if (i == j && sx == 0 && sy == 0 && sz == 0) {
              continue;
            }

            const double dx = rj[0] + sx * a[0] + sy * b[0] + sz * c[0] - ri[0];
            const double dy = rj[1] + sx * a[1] + sy * b[1] + sz * c[1] - ri[1];
            const double dz = rj[2] + sx * a[2] + sy * b[2] + sz * c[2] - ri[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < tiny) {
              continue;
            }

            if (r2 < rc_radial2) {
              neighbors_radial.push_back({i, j, sx, sy, sz, dx, dy, dz, std::sqrt(r2)});
            }
            if (r2 < rc_angular2) {
              neighbors_angular.push_back({i, j, sx, sy, sz, dx, dy, dz, std::sqrt(r2)});
            }
            if (r2 < rc_hopping2 && canonical_bond(i, j, sx, sy, sz)) {
              hopping_bonds.push_back({i, j, sx, sy, sz, dx, dy, dz, std::sqrt(r2)});
            }
          }
        }
      }
    }
  }
}

void DatasetCollection::print_summary() const
{
  std::size_t total_atoms = 0, total_kpoints = 0;
  for (const auto& data : structures_) {
    total_atoms += data.structure.num_atoms;
    total_kpoints += data.kpoints.size();
  }
  print_line_1();
  printf("Dataset summary\n");
  print_line_2();
  printf("  number of structures          = %zu\n", structures_.size());
  printf("  total atoms                   = %zu\n", total_atoms);
  printf("  total k-points                = %zu\n", total_kpoints);
}

DatasetCollection::Batch::Batch(const Parameters& para, const DatasetCollection& dataset)
  : structure_rng_(static_cast<unsigned int>(para.seed) ^ 0x6a09e667U)
{
  const int num_structures = dataset.size();
  if (num_structures == 0 || para.structure_batch <= 0 || para.k_batch <= 0) {
    PRINT_INPUT_ERROR("Batch requires data and positive batch sizes.");
  }
  structure_permutation_.resize(num_structures);
  std::iota(structure_permutation_.begin(), structure_permutation_.end(), 0);
  std::shuffle(structure_permutation_.begin(), structure_permutation_.end(), structure_rng_);
  structure_cursor_ = 0;
  structure_indices_.resize(std::min(para.structure_batch, num_structures));

  k_permutations_.resize(num_structures);
  k_cursors_.assign(num_structures, 0);
  k_indices_.resize(num_structures);
  k_rngs_.reserve(num_structures);

  for (int s = 0; s < num_structures; ++s) {
    const int nk = static_cast<int>(dataset[s].kpoints.size());
    if (nk <= 0) {
      PRINT_INPUT_ERROR("Every dataset structure requires at least one k-point.");
    }

    auto& permutation = k_permutations_[s];
    permutation.resize(nk);
    std::iota(permutation.begin(), permutation.end(), 0);

    const unsigned int k_seed =
      static_cast<unsigned int>(para.seed) + static_cast<unsigned int>(s) * 0x9e3779b9U;
    k_rngs_.emplace_back(k_seed);
    std::shuffle(permutation.begin(), permutation.end(), k_rngs_.back());

    k_indices_[s].resize(std::min(para.k_batch, nk));
  }
}

void DatasetCollection::Batch::next()
{
  draw_batch(
    structure_permutation_,
    structure_cursor_,
    static_cast<int>(structure_indices_.size()),
    structure_rng_,
    structure_indices_);
  for (int s : structure_indices_) {
    draw_batch(
      k_permutations_[s],
      k_cursors_[s],
      static_cast<int>(k_indices_[s].size()),
      k_rngs_[s],
      k_indices_[s]);
  }
}

int DatasetCollection::Batch::estimated_steps_per_epoch() const
{
  double total_k = 0.0;
  double mean_k_per_selected_structure = 0.0;
  for (std::size_t s = 0; s < k_permutations_.size(); ++s) {
    total_k += static_cast<double>(k_permutations_[s].size());
    mean_k_per_selected_structure += static_cast<double>(k_indices_[s].size());
  }
  mean_k_per_selected_structure /= static_cast<double>(k_permutations_.size());
  const double samples_per_step =
    static_cast<double>(structure_indices_.size()) * mean_k_per_selected_structure;
  return std::max(1, static_cast<int>(std::ceil(total_k / samples_per_step)));
}
