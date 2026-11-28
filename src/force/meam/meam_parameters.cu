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
Major contributors:
  OpenAI Codex: Initial implementation.
------------------------------------------------------------------------------*/

#include "meam_functions.cuh"
#include "meam_parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <array>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace
{
struct Lattice {
  int first;
  int second;
  int screeners;
  double ratio;
  std::array<double, 3> shape;
  bool distinct_shells;
  double screening_geometry = -1.0;
};

const std::map<std::string, Lattice> lattices{
  {"fcc", {12, 6, 4, sqrt(2.0), {}, false}},
  {"bcc", {8, 6, 4, 2.0 / sqrt(3.0), {}, false}},
  {"hcp", {12, 6, 4, sqrt(2.0), {0.0, 0.0, 1.0 / 3.0}, false}},
  {"dia", {4, 12, 1, sqrt(8.0 / 3.0), {0.0, 0.0, 32.0 / 9.0}, true}},
  {"dia3", {4, 12, 4, sqrt(11.0 / 3.0), {0.0, 0.0, 32.0 / 9.0}, false, 1.0}},
  {"dim", {1, 0, 0, 1.0, {1.0, 2.0 / 3.0, 0.4}, false}},
  {"b1", {6, 12, 2, sqrt(2.0), {}, true}},
  {"b2", {8, 6, 4, 2.0 / sqrt(3.0), {}, true}},
  {"sc", {6, 12, 2, sqrt(2.0), {}, false}},
  {"c11", {10, 0, 0, 1.0, {}, false}},
  {"l12", {12, 6, 4, sqrt(2.0), {}, true, 1.0}},
  {"ch4", {4, 0, 0, 1.0, {0.0, 0.0, 32.0 / 9.0}, false}},
  {"lin", {2, 0, 0, 1.0, {0.0, 8.0 / 3.0, 0.0}, false}},
  {"zig", {2, 2, 1, 2.0, {}, false}},
  {"tri", {2, 1, 2, 2.0, {}, false}},
};

struct PairInput {
  std::string lattice = "fcc";
  double re = 0.0;
  double cohesive = 0.0;
  double alpha = 0.0;
  double delta = 0.0;
  double repulsion = 0.0;
  double attraction = 0.0;
  double half_angle = 1.5707963267948966;
  bool second_neighbors = false;
  bool zbl = true;
};

struct ScreenInput {
  double minimum = 2.0;
  double maximum = 2.8;
};

[[noreturn]] void input_error(const std::string& file, const std::string& reason)
{
  const std::string message = "MEAM " + file + ": " + reason;
  PRINT_INPUT_ERROR(message.c_str());
}

double real(const std::string& token, const std::string& file, const std::string& field)
{
  double value;
  if (!is_valid_real(token, &value) || !std::isfinite(value)) {
    input_error(file, "invalid " + field + " '" + token + "'");
  }
  return value;
}

int integer(const std::string& token, const std::string& file, const std::string& field)
{
  int value;
  if (!is_valid_int(token, &value)) {
    input_error(file, "invalid " + field + " '" + token + "'");
  }
  return value;
}

std::string trim(const std::string& text)
{
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

std::string unquote(std::string text)
{
  if (
    text.size() >= 2 && ((text.front() == '\'' && text.back() == '\'') ||
                         (text.front() == '"' && text.back() == '"'))) {
    return text.substr(1, text.size() - 2);
  }
  return text;
}

std::string resolve_path(const std::string& descriptor, const std::string& name)
{
  const std::string path = unquote(name);
  if (
    (!path.empty() && (path.front() == '/' || path.front() == '\\')) ||
    (path.size() >= 2 && path[1] == ':')) {
    return path;
  }
  const auto slash = descriptor.find_last_of("/\\");
  return (slash == std::string::npos ? std::string() : descriptor.substr(0, slash + 1)) + path;
}

const Lattice& lattice(const std::string& name, const std::string& filename)
{
  const auto it = lattices.find(name);
  if (it == lattices.end()) {
    input_error(filename, "unsupported reference lattice '" + name + "'");
  }
  return it->second;
}

Lattice reference_geometry(const PairInput& pair)
{
  Lattice geometry = lattices.at(pair.lattice);
  if (pair.lattice == "zig" || pair.lattice == "tri") {
    const double s = sin(pair.half_angle);
    const double c = cos(pair.half_angle);
    const double ss = s * s;
    const double cc = c * c;
    geometry.ratio = 2.0 * s;
    geometry.shape = {
      4.0 * cc, 4.0 * (cc * cc + ss * ss - 1.0 / 3.0), 4.0 * cc * (3.0 * ss * ss + cc * cc - 0.6)};
  }
  return geometry;
}

std::vector<std::string> records(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input) {
    input_error(filename, "cannot open file");
  }
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    line = trim(line.substr(0, line.find('#')));
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  if (input.bad()) {
    input_error(filename, "failed while reading file");
  }
  return lines;
}

void read_library(
  const std::string& filename,
  const std::vector<std::string>& names,
  std::vector<MeamElement>& elements,
  std::vector<int>& atomic_numbers,
  std::vector<PairInput>& pairs)
{
  const int n = static_cast<int>(names.size());
  std::vector<std::string> tokens;
  for (const auto& line : records(filename)) {
    const auto row = get_tokens(line);
    tokens.insert(tokens.end(), row.begin(), row.end());
  }
  constexpr size_t fields = 19;
  if (tokens.size() % fields != 0) {
    input_error(filename, "truncated library record (expected 19 fields per element)");
  }
  for (int i = 0; i < n; ++i) {
    size_t offset = 0;
    while (offset < tokens.size() && unquote(tokens[offset]) != names[i]) {
      offset += fields;
    }
    if (offset == tokens.size()) {
      input_error(filename, "element '" + names[i] + "' is missing");
    }
    auto& element = elements[i];
    auto& pair = pairs[i * n + i];
    const auto number = [&](int k, const char* field) {
      return real(tokens[offset + k], filename, names[i] + " " + field);
    };
    const auto whole = [&](int k, const char* field) {
      return integer(tokens[offset + k], filename, names[i] + " " + field);
    };
    pair.lattice = unquote(tokens[offset + 1]);
    const auto& geometry = lattice(pair.lattice, filename);
    if (number(2, "coordination") <= 0.0 || number(4, "mass") <= 0.0) {
      input_error(filename, "coordination and mass must be positive for " + names[i]);
    }
    atomic_numbers[i] = whole(3, "atomic number");
    if (atomic_numbers[i] < 0 || atomic_numbers[i] > 118) {
      input_error(filename, "invalid atomic number for " + names[i]);
    }
    pair.alpha = number(5, "alpha");
    for (int rank = 0; rank < 4; ++rank) {
      element.beta[rank] = number(6 + rank, "beta");
    }
    pair.re = number(10, "lattice constant");
    if (pair.lattice == "fcc") {
      pair.re /= sqrt(2.0);
    } else if (pair.lattice == "bcc") {
      pair.re *= sqrt(3.0) / 2.0;
    } else if (pair.lattice == "dia" || pair.lattice == "dia3") {
      pair.re *= sqrt(3.0) / 4.0;
    }
    pair.cohesive = number(11, "cohesive energy");
    element.embedding_scale = number(12, "embedding prefactor");
    if (number(13, "t0") != 1.0) {
      input_error(filename, "t0 must be 1 for " + names[i]);
    }
    for (int rank = 0; rank < 3; ++rank) {
      element.t[rank] = number(14 + rank, "angular weight");
    }
    element.density_scale = number(17, "density scale");
    element.density_function = whole(18, "ibar");
    if (
      element.density_function != -5 && element.density_function != 0 &&
      element.density_function != 1 && element.density_function != 3 &&
      element.density_function != 4) {
      input_error(filename, "unsupported ibar for " + names[i]);
    }
    element.coordination = geometry.first;
    if (element.density_scale <= 0.0) {
      input_error(filename, "density scale must be positive for " + names[i]);
    }
  }
}

int read_overrides(
  const std::string& file,
  MeamParameters& model,
  std::vector<PairInput>& pairs,
  std::vector<ScreenInput>& screening)
{
  auto& settings = model.settings;
  const int n = settings.type_count;
  bool augment_first_weight = true;
  int rose_form = 0;
  if (!file.empty()) {
    for (const auto& line : records(file)) {
      const auto eq = line.find('=');
      if (eq == std::string::npos) {
        input_error(file, "missing '=' in '" + line + "'");
      }
      const std::string left = trim(line.substr(0, eq));
      const std::string value = unquote(trim(line.substr(eq + 1)));
      const auto opening = left.find('(');
      const std::string key = trim(left.substr(0, opening));
      std::vector<int> indices;
      if (opening != std::string::npos) {
        if (left.back() != ')') {
          input_error(file, "unclosed indices in '" + line + "'");
        }
        std::istringstream stream(left.substr(opening + 1, left.size() - opening - 2));
        for (std::string component; std::getline(stream, component, ',');) {
          const int index = integer(trim(component), file, key + " index");
          if (index < 1 || index > n) {
            input_error(file, "element index out of range in '" + line + "'");
          }
          indices.push_back(index - 1);
        }
      }
      const auto count = [&](size_t expected) {
        if (indices.size() != expected || (opening != std::string::npos) != (expected != 0)) {
          input_error(file, "wrong number of indices in '" + line + "'");
        }
      };
      const auto scalar = [&] { return real(value, file, key); };
      const auto option = [&](int maximum) {
        const int choice = integer(value, file, key);
        if (choice < 0 || choice > maximum) {
          input_error(file, "invalid value for " + key);
        }
        return choice;
      };
      if (key == "rc" || key == "delr" || key == "gsmooth_factor") {
        count(0);
        double& field = key == "rc"
                          ? settings.cutoff
                          : (key == "delr" ? settings.smoothing_width : settings.gamma_smoothing);
        field = scalar();
      } else if (
        key == "augt1" || key == "ialloy" || key == "erose_form" || key == "mixture_ref_t" ||
        key == "bkgd_dyn" || key == "emb_lin_neg") {
        count(0);
        if (key == "ialloy") {
          settings.alloy_rule = option(2);
        } else if (key == "erose_form") {
          rose_form = option(2);
        } else if (key == "augt1") {
          augment_first_weight = option(1);
        } else if (key == "mixture_ref_t") {
          settings.mixed_reference = option(1);
        } else if (key == "bkgd_dyn") {
          settings.dynamic_background = option(1);
        } else {
          settings.linear_negative_embedding = option(1);
        }
      } else if (key == "rho0") {
        count(1);
        model.elements[indices[0]].density_scale = scalar();
      } else if (key == "Cmin" || key == "Cmax") {
        count(3);
        const auto a = std::min(indices[0], indices[1]);
        const auto b = std::max(indices[0], indices[1]);
        auto& screen = screening[(a * n + b) * n + indices[2]];
        double& field = key == "Cmin" ? screen.minimum : screen.maximum;
        field = scalar();
      } else if (
        key == "Ec" || key == "delta" || key == "re" || key == "alpha" || key == "attrac" ||
        key == "repuls" || key == "nn2" || key == "zbl" || key == "lattce" || key == "theta") {
        count(2);
        const auto a = std::min(indices[0], indices[1]);
        const auto b = std::max(indices[0], indices[1]);
        auto& pair = pairs[a * n + b];
        if (key == "lattce") {
          lattice(value, file);
          pair.lattice = value;
        } else if (key == "nn2") {
          pair.second_neighbors = option(1);
        } else if (key == "zbl") {
          pair.zbl = option(1);
        } else if (key == "theta") {
          const double degrees = scalar();
          if (degrees <= 0.0 || degrees > 180.0) {
            input_error(file, "theta must be in (0, 180] degrees");
          }
          pair.half_angle = degrees * (1.5707963267948966 / 180.0);
        } else {
          const std::map<std::string, double PairInput::*> properties{
            {"Ec", &PairInput::cohesive},
            {"delta", &PairInput::delta},
            {"re", &PairInput::re},
            {"alpha", &PairInput::alpha},
            {"attrac", &PairInput::attraction},
            {"repuls", &PairInput::repulsion}};
          pair.*properties.at(key) = scalar();
        }
      } else {
        input_error(file, "unsupported parameter '" + key + "'");
      }
    }
  }
  if (
    settings.cutoff <= 0.0 || settings.smoothing_width <= 0.0 ||
    settings.smoothing_width > settings.cutoff || settings.gamma_smoothing <= 0.0) {
    input_error(file, "invalid cutoff or smoothing width");
  }
  for (int i = 0; i < n; ++i) {
    auto& element = model.elements[i];
    const auto& pure = pairs[i * n + i];
    const auto geometry = reference_geometry(pure);
    element.coordination = geometry.first;
    std::copy(geometry.shape.begin(), geometry.shape.end(), element.reference_shape);
    element.equilibrium_distance = pure.re;
    element.embedding_scale *= pure.cohesive;
    if (
      element.density_scale <= 0.0 || pure.re <= 0.0 || pure.alpha <= 0.0 || pure.cohesive <= 0.0) {
      input_error(file, "invalid pure-element parameters for element " + std::to_string(i + 1));
    }
    if (augment_first_weight) {
      element.t[0] += 0.6 * element.t[2];
    }
  }
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      auto& pair = pairs[i * n + j];
      const auto& a = pairs[i * n + i];
      const auto& b = pairs[j * n + j];
      if (i != j) {
        if (pair.re == 0.0) {
          pair.re = 0.5 * (a.re + b.re);
        }
        if (pair.cohesive == 0.0) {
          double fraction = 0.5;
          if (pair.lattice == "l12") {
            fraction = 0.75;
          } else if (pair.lattice == "c11") {
            fraction = a.lattice == "dia" ? 2.0 / 3.0 : 1.0 / 3.0;
          }
          pair.cohesive = fraction * a.cohesive + (1.0 - fraction) * b.cohesive - pair.delta;
        }
        if (pair.alpha == 0.0) {
          pair.alpha = 0.5 * (a.alpha + b.alpha);
        }
        pairs[j * n + i] = pair;
      }
      if (pair.re <= 0.0 || pair.cohesive <= 0.0 || pair.alpha <= 0.0) {
        input_error(file, "invalid reference parameters for an element pair");
      }
      for (int k = 0; k < n; ++k) {
        const auto& screen = screening[(i * n + j) * n + k];
        if (screen.maximum <= screen.minimum) {
          input_error(file, "Cmax must be greater than Cmin");
        }
        screening[(j * n + i) * n + k] = screen;
      }
    }
  }
  settings.inverse_table_step = MEAM_TABLE_POINTS / (1.1 * settings.cutoff);
  return rose_form;
}

double atomic_density(const MeamElement& element, double r, int rank)
{
  return element.density_scale *
         exp(-element.beta[rank] * (r / element.equilibrium_distance - 1.0));
}

class ReferenceModel
{
public:
  ReferenceModel(
    MeamParameters& model,
    const std::vector<PairInput>& pairs,
    const std::vector<ScreenInput>& screening,
    const std::vector<int>& atomic_numbers,
    int rose_form)
    : model_(model),
      pairs_(pairs),
      screening_(screening),
      atomic_numbers_(atomic_numbers),
      rose_form_(rose_form),
      n_(model.settings.type_count)
  {
  }

  void initialize(const std::string& filename)
  {
    for (int i = 0; i < n_; ++i) {
      auto& element = model_.elements[i];
      const auto& pair = pairs_[i * n_ + i];
      const auto geometry = reference_geometry(pair);
      double density = geometry.first * element.density_scale;
      if (pair.second_neighbors) {
        density += second_shell_weight(geometry, i, i) *
                   atomic_density(element, geometry.ratio * pair.re, 0);
      }
      element.reference_density = density * reference_scale(element, element.t);
      if (!std::isfinite(element.reference_density) || element.reference_density <= 0.0) {
        input_error(filename, "invalid background density for element " + std::to_string(i + 1));
      }
    }

    model_.pair_tables.resize(n_ * n_ * MEAM_TABLE_POINTS);
    std::array<double, MEAM_TABLE_POINTS> values;
    std::array<double, MEAM_TABLE_POINTS> slopes;
    for (int i = 0; i < n_; ++i) {
      for (int j = i; j < n_; ++j) {
        for (int k = 0; k < MEAM_TABLE_POINTS; ++k) {
          const double r = k / model_.settings.inverse_table_step;
          values[k] = pair_energy(i, j, r);
          if (!std::isfinite(values[k])) {
            input_error(filename, "non-finite pair energy in radial table");
          }
        }
        for (int k = 0; k < MEAM_TABLE_POINTS; ++k) {
          if (k == 0) {
            slopes[k] = values[1] - values[0];
          } else if (k == MEAM_TABLE_POINTS - 1) {
            slopes[k] = 0.0;
          } else if (k == 1 || k == MEAM_TABLE_POINTS - 2) {
            slopes[k] = 0.5 * (values[k + 1] - values[k - 1]);
          } else {
            slopes[k] =
              (8.0 * (values[k + 1] - values[k - 1]) - (values[k + 2] - values[k - 2])) / 12.0;
          }
        }
        for (int k = 0; k < MEAM_TABLE_POINTS - 1; ++k) {
          const double difference = values[k + 1] - values[k];
          const MeamSpline spline{
            values[k],
            slopes[k],
            3.0 * difference - 2.0 * slopes[k] - slopes[k + 1],
            slopes[k] + slopes[k + 1] - 2.0 * difference};
          model_.pair_tables[(i * n_ + j) * MEAM_TABLE_POINTS + k] = spline;
          model_.pair_tables[(j * n_ + i) * MEAM_TABLE_POINTS + k] = spline;
        }
      }
    }
  }

private:
  const MeamSettings& settings() const { return model_.settings; }

  double reference_scale(const MeamElement& element, const double* weights) const
  {
    if (element.density_function <= 0) {
      return 1.0;
    }
    double gamma = 0.0;
    for (int k = 0; k < 3; ++k) {
      gamma += weights[k] * element.reference_shape[k];
    }
    gamma /= element.coordination * element.coordination;
    return meam_angular_scaling(gamma, element.density_function, settings().gamma_smoothing).value;
  }

  double second_shell_weight(const Lattice& geometry, int same, int other) const
  {
    if (geometry.second == 0) {
      return 0.0;
    }
    const double c = geometry.screening_geometry >= 0.0
                       ? geometry.screening_geometry
                       : 4.0 / (geometry.ratio * geometry.ratio) - 1.0;
    return geometry.second * pow(reference_screen(same, other, c), geometry.screeners);
  }

  double reference_screen(int center, int screener, double geometry) const
  {
    const auto& screen = screening_[(center * n_ + center) * n_ + screener];
    return meam_cutoff((geometry - screen.minimum) / (screen.maximum - screen.minimum)).value;
  }

  double l12_shell(int i, int j, bool first) const
  {
    const double other = reference_screen(i, j, 1.0);
    return 6.0 * (first ? pow(reference_screen(i, i, 1.0) * other, 2) : pow(other, 4));
  }

  double reference_embedding(int i, int j, const PairInput& pair, double r, bool first) const
  {
    const auto& central = model_.elements[i];
    const auto& neighbor = model_.elements[j];
    const auto geometry = reference_geometry(pair);
    std::array<double, 3> shape = geometry.shape;
    int bonds = geometry.first;
    if (pair.lattice == "ch4" && !first) {
      bonds = 1;
      shape = lattices.at("dim").shape;
    } else if (pair.lattice == "tri" && first) {
      bonds = 1;
      shape = lattices.at("dim").shape;
    }
    const double from_neighbor = atomic_density(neighbor, r, 0);
    const double from_center = atomic_density(central, r, 0);
    double density = bonds * from_neighbor;
    if (pair.lattice == "l12" && first) {
      density = 8.0 * from_center + 4.0 * from_neighbor;
    }
    const double first_shell = density;
    if (pair.second_neighbors) {
      const double second_shell =
        pair.lattice == "l12" ? l12_shell(i, j, first) : second_shell_weight(geometry, i, j);
      density += second_shell * atomic_density(central, r * geometry.ratio, 0);
    }
    if (density < 1e-14) {
      return 0.0;
    }
    double gamma = 0.0;
    double weights[3];
    for (int k = 0; k < 3; ++k) {
      weights[k] = settings().alloy_rule == 2 ? central.t[k] : neighbor.t[k];
      if (pair.lattice == "l12" && first && settings().alloy_rule != 2 && first_shell > 0.0) {
        weights[k] =
          (8.0 * from_center * central.t[k] + 4.0 * from_neighbor * neighbor.t[k]) / first_shell;
      }
      if (shape[k] != 0.0) {
        const double scaled = atomic_density(neighbor, r, k + 1) / density;
        gamma += weights[k] * shape[k] * scaled * scaled;
      }
    }
    if (pair.lattice == "l12" && first) {
      double u = atomic_density(central, r, 2);
      double v = atomic_density(neighbor, r, 2);
      double scale = 1.0;
      if (settings().alloy_rule == 1) {
        u *= central.t[1];
        v *= neighbor.t[1];
        const double denominator = 8.0 * from_center * central.t[1] * central.t[1] +
                                   4.0 * from_neighbor * neighbor.t[1] * neighbor.t[1];
        if (denominator > 0.0) {
          scale = first_shell / denominator;
        }
      }
      const double contrast = (u - v) / density;
      gamma += (8.0 / 3.0) * weights[1] * contrast * contrast * scale;
    }
    const double g =
      meam_angular_scaling(gamma, central.density_function, settings().gamma_smoothing).value;
    double background = central.reference_density;
    if (settings().mixed_reference || settings().dynamic_background) {
      background = central.density_scale * central.coordination;
      if (settings().mixed_reference) {
        background *= reference_scale(central, weights);
      }
    }
    return meam_embedding(
             density * g / background,
             central.embedding_scale,
             settings().linear_negative_embedding)
      .value;
  }

  double c11_pair(int i, int j, double r, double rose) const
  {
    const int a = pairs_[i * n_ + i].lattice == "dia" ? i : j;
    const int b = a == i ? j : i;
    const auto& first = model_.elements[a];
    const auto& second = model_.elements[b];
    double p[4], q[4], t[3];
    for (int k = 0; k < 4; ++k) {
      p[k] = atomic_density(first, r, k);
      q[k] = atomic_density(second, r, k);
    }
    for (int k = 0; k < 3; ++k) {
      t[k] = settings().alloy_rule == 2 ? first.t[k]
                                        : (p[0] * first.t[k] + q[0] * second.t[k]) / (p[0] + q[0]);
    }
    const double density_a = sqrt(
      pow(5.0 * (p[0] + q[0]), 2) + t[0] * pow(q[1] - p[1], 2) +
      (t[1] / 6.0) * pow(q[2] + p[2], 2) + (121.0 / 40.0) * t[2] * pow(q[3] - p[3], 2));
    const double density_b = sqrt(100.0 * p[0] * p[0] + (2.0 / 3.0) * t[1] * p[2] * p[2]);
    const double fa =
      meam_embedding(density_a, first.embedding_scale, settings().linear_negative_embedding).value;
    const double fb =
      meam_embedding(density_b, second.embedding_scale, settings().linear_negative_embedding).value;
    return (3.0 * rose - 2.0 * fa - fb - 5.0 * nearest_pair(a, a, r)) / 10.0;
  }

  double nearest_pair(int i, int j, double r) const
  {
    if (r == 0.0) {
      return 0.0;
    }
    const auto& pair = pairs_[i * n_ + j];
    if (
      atomic_density(model_.elements[i], r, 0) <= 1e-14 &&
      atomic_density(model_.elements[j], r, 0) <= 1e-14) {
      return 0.0;
    }
    const double strain = pair.alpha * (r / pair.re - 1.0);
    double cubic = strain < 0.0 ? pair.repulsion : pair.attraction;
    if (rose_form_ == 0) {
      cubic *= pair.re / r;
    } else if (rose_form_ == 1) {
      cubic = -pair.attraction + pair.repulsion / r;
    }
    const double rose =
      -pair.cohesive * (1.0 + strain + cubic * strain * strain * strain) * exp(-strain);
    if (pair.lattice == "c11" && i != j) {
      return c11_pair(i, j, r, rose);
    }
    const double fa = reference_embedding(i, j, pair, r, true);
    const double fb = reference_embedding(j, i, pair, r, false);
    if (pair.lattice == "l12" && i != j) {
      return rose / 3.0 - fa / 4.0 - fb / 12.0 - second_neighbor_series(i, i, r);
    }
    if (pair.lattice == "ch4") {
      return (5.0 * rose - fa - 4.0 * fb) / 4.0;
    }
    if (pair.lattice == "zig" || pair.lattice == "tri") {
      const bool chain = pair.lattice == "zig";
      const double r2 = r * reference_geometry(pair).ratio;
      double result = chain ? 2.0 * rose - fa - fb : 3.0 * rose - 2.0 * fa - fb;
      if (i != j) {
        result -= reference_screen(i, j, 1.0) * nearest_pair(i, i, r2);
        if (chain) {
          result -= reference_screen(j, i, 1.0) * nearest_pair(j, j, r2);
        }
      }
      return 0.5 * result;
    }
    return (2.0 * rose - fa - fb) / lattices.at(pair.lattice).first;
  }

  double second_neighbor_series(int i, int j, double r) const
  {
    const auto geometry = reference_geometry(pairs_[i * n_ + j]);
    const double factor = -second_shell_weight(geometry, i, j) / geometry.first;
    double value = nearest_pair(i, j, r);
    double coefficient = factor;
    for (int term = 0; term < 10 && coefficient != 0.0; ++term) {
      r *= geometry.ratio;
      const double correction = coefficient * nearest_pair(i, j, r);
      value += correction;
      if (fabs(correction) < 1e-20) {
        break;
      }
      coefficient *= factor;
    }
    return value;
  }

  double pair_energy(int i, int j, double r) const
  {
    const auto& pair = pairs_[i * n_ + j];
    const auto geometry = reference_geometry(pair);
    double result = nearest_pair(i, j, r);
    if (pair.second_neighbors) {
      if (geometry.distinct_shells) {
        const double r2 = r * geometry.ratio;
        if (pair.lattice == "l12") {
          result -= 0.75 * (l12_shell(i, j, true) / 6.0) * second_neighbor_series(i, i, r2) +
                    0.25 * (l12_shell(j, i, false) / 6.0) * second_neighbor_series(j, j, r2);
        } else {
          result -= (second_shell_weight(geometry, i, j) * second_neighbor_series(i, i, r2) +
                     second_shell_weight(geometry, j, i) * second_neighbor_series(j, j, r2)) /
                    (2.0 * geometry.first);
        }
      } else {
        result = second_neighbor_series(i, j, r);
      }
    }
    const int a = atomic_numbers_[i];
    const int b = atomic_numbers_[j];
    const double strain = pair.alpha * (r / pair.re - 1.0);
    if (pair.zbl && strain < -1.0) {
      const double x = r * (pow(a, 0.23) + pow(b, 0.23)) / 0.4685;
      double zbl = 0.028171 * exp(-0.20162 * x) + 0.28022 * exp(-0.40290 * x) +
                   0.50986 * exp(-0.94229 * x) + 0.18175 * exp(-3.1998 * x);
      if (r > 0.0) {
        zbl *= 14.3997 * a * b / r;
      }
      const double weight = meam_cutoff(0.5 * (strain + 3.0)).value;
      result = weight * result + (1.0 - weight) * zbl;
    }
    return result;
  }

  MeamParameters& model_;
  const std::vector<PairInput>& pairs_;
  const std::vector<ScreenInput>& screening_;
  const std::vector<int>& atomic_numbers_;
  const int rose_form_;
  const int n_;
};
} // namespace

MeamParameters read_meam_parameters(const char* filename, int number_of_types)
{
  const auto input = records(filename);
  if (number_of_types < 1 || number_of_types > 32 || input.size() < 3 || input.size() > 4) {
    input_error(
      filename,
      "expected a header (1-32 elements), library and parameter paths, "
      "and an optional library_elements line");
  }
  auto header = get_tokens(input[0]);
  const size_t n = number_of_types;
  if (
    header.size() != n + 2 || header[0] != "meam" ||
    integer(header[1], filename, "type count") != number_of_types) {
    input_error(filename, "expected 'meam number_of_types element1 ...' on the first line");
  }
  const std::vector<std::string> names(header.begin() + 2, header.end());
  auto sorted = names;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    input_error(filename, "the element list contains duplicates");
  }
  const auto library = resolve_path(filename, input[1]);
  const auto params = input[2] == "NULL" ? std::string() : resolve_path(filename, input[2]);
  auto library_names = names;
  if (input.size() == 4) {
    const auto aliases = get_tokens(input[3]);
    if (aliases.size() != n + 1 || aliases.front() != "library_elements") {
      input_error(filename, "expected one library_elements name per header element");
    }
    library_names.assign(aliases.begin() + 1, aliases.end());
  }

  MeamParameters model;
  model.settings.type_count = number_of_types;
  model.elements.resize(n);
  std::vector<PairInput> pairs(n * n);
  std::vector<ScreenInput> screen_inputs(n * n * n);
  std::vector<int> atomic_numbers(n);
  read_library(library, library_names, model.elements, atomic_numbers, pairs);
  const int rose_form = read_overrides(params, model, pairs, screen_inputs);
  model.screening.resize(n * n * n);
  model.screening_bounds.resize(n * n);
  for (size_t pair = 0; pair < n * n; ++pair) {
    double bound = 1.0;
    for (size_t k = 0; k < n; ++k) {
      const size_t index = pair * n + k;
      const auto& input_screen = screen_inputs[index];
      const double c = input_screen.maximum;
      // Within the screening ellipse the maximum squared radius is 1 below Cmax=2.
      if (c > 2.0) {
        bound = std::max(bound, c * c / (4.0 * (c - 1.0)));
      }
      model.screening[index] = {input_screen.minimum, 1.0 / (c - input_screen.minimum)};
    }
    model.screening_bounds[pair] = bound;
  }
  ReferenceModel(model, pairs, screen_inputs, atomic_numbers, rose_form).initialize(filename);
  return model;
}
