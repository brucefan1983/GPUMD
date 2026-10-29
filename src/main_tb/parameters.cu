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

#include "parameters.cuh"
#include "utilities/error.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace
{
template <typename T>
T read_number(const std::string& token, const std::string& context)
{
  std::istringstream input(token);
  T value;
  if (!(input >> value) || input.peek() != std::char_traits<char>::eof() || !std::isfinite(value)) {
    PRINT_INPUT_ERROR(("Invalid number '" + token + "' " + context + ".").c_str());
  }
  return value;
}
} // namespace

Parameters::Parameters(const std::string& input_file)
{
  read_input(input_file);
  validate();
  derive_model_sizes();
  derive_band_weights();
}

void Parameters::read_input(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR(("Failed to open " + filename + ".").c_str());
  }

  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto tokens = get_tokens_without_comments(line);
    if (!tokens.empty()) {
      parse_keyword(tokens, filename + ":" + std::to_string(line_number));
    }
  }
}

void Parameters::parse_keyword(const std::vector<std::string>& tokens, const std::string& location)
{
  const std::string& keyword = tokens[0];
  const std::string context = "for keyword '" + keyword + "' at " + location;
  std::size_t position = 1;
  auto next = [&]() -> const std::string& {
    if (position == tokens.size()) {
      PRINT_INPUT_ERROR(("Missing value " + context + ".").c_str());
    }
    return tokens[position++];
  };
  auto read_int = [&]() { return read_number<int>(next(), context); };
  auto read_double = [&]() { return read_number<double>(next(), context); };

  if (keyword == "type") {
    if (!elements.empty()) {
      PRINT_INPUT_ERROR(
        ("The type keyword can only be specified once (" + location + ").").c_str());
    }
    num_types = read_int();
    if (num_types <= 0 || tokens.size() - position != static_cast<std::size_t>(num_types)) {
      PRINT_INPUT_ERROR(
        ("Expected type N symbol1 ... symbolN with positive N at " + location + ".").c_str());
    }
    elements.assign(tokens.begin() + position, tokens.end());
    return;
  }

  if (keyword == "prediction") {
    prediction = read_int();
  } else if (keyword == "basis") {
    basis = next();
  } else if (keyword == "cutoff") {
    rc_radial = read_double();
    rc_angular = read_double();
    rc_hopping = read_double();
  } else if (keyword == "n_max") {
    n_max_radial = read_int();
    n_max_angular = read_int();
  } else if (keyword == "basis_size") {
    basis_size_radial = read_int();
    basis_size_angular = read_int();
  } else if (keyword == "l_max") {
    L_max = read_int();
  } else if (keyword == "neuron") {
    num_neurons_onsite = read_int();
    num_neurons_hopping = read_int();
  } else if (keyword == "epoch") {
    maximum_epoch = read_int();
  } else if (keyword == "seed") {
    seed = read_int();
  } else if (keyword == "learning_rate") {
    learning_rate = read_double();
  } else if (keyword == "weight_decay") {
    weight_decay = read_double();
  } else if (keyword == "lr_scheduler") {
    if (next() != "plateau") {
      PRINT_INPUT_ERROR(("Only lr_scheduler plateau is supported (" + location + ").").c_str());
    }
    lr_scheduler_explicit = true;
    if (position != tokens.size()) {
      lr_factor = read_double();
      lr_patience = read_int();
      lr_rel_threshold = read_double();
      lr_min = read_double();
    }
  } else if (keyword == "batch") {
    structure_batch = read_int();
    k_batch = read_int();
  } else if (keyword == "band_range") {
    band_min = read_int();
    band_max = read_int();
  } else if (keyword == "band_curriculum") {
    band_curriculum = read_int();
  } else if (keyword == "band_weight") {
    BandWeightRange range;
    range.band_min = read_int();
    range.band_max = read_int();
    range.weight = read_double();
    band_weight_ranges_.push_back(range);
  } else if (keyword == "dispersion") {
    dispersion_weight = read_double();
    const auto& token = next();
    if (token == "inf" || token == "infinity" || token == "INF" || token == "INFINITY") {
      dispersion_sigma = std::numeric_limits<double>::infinity();
    } else {
      dispersion_sigma = read_number<double>(token, context);
    }

  } else {
    PRINT_INPUT_ERROR(("Unknown keyword '" + keyword + "' at " + location + ".").c_str());
  }

  if (position != tokens.size()) {
    PRINT_INPUT_ERROR(("Too many values " + context + ".").c_str());
  }
}

void Parameters::validate() const
{
  if (prediction != 0 && prediction != 1) {
    PRINT_INPUT_ERROR("prediction must be 0 or 1.");
  }
  if (elements.empty()) {
    PRINT_INPUT_ERROR("The type keyword is required: type N symbol1 ... symbolN.");
  }
  for (std::size_t i = 0; i < elements.size(); ++i) {
    if (elements[i].empty()) {
      PRINT_INPUT_ERROR("Element symbols cannot be empty.");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (elements[i] == elements[j]) {
        PRINT_INPUT_ERROR(("Duplicate element symbol '" + elements[i] + "'.").c_str());
      }
    }
  }
  if (basis != "sp3" && basis != "sp3sstar" && basis != "sp3d5" && basis != "sp3d5sstar") {
    PRINT_INPUT_ERROR("basis must be one of: sp3, sp3sstar, sp3d5, sp3d5sstar.");
  }
  if (rc_radial <= 0.0 || rc_angular <= 0.0 || rc_hopping <= 0.0) {
    PRINT_INPUT_ERROR("All cutoff radii must be positive.");
  }
  for (double cutoff : {rc_radial, rc_angular, rc_hopping}) {
    if (cutoff < std::numeric_limits<float>::min() || cutoff > std::numeric_limits<float>::max()) {
      PRINT_INPUT_ERROR("Cutoff radii are outside the single-precision GPU range.");
    }
  }
  if (rc_hopping > rc_radial) {
    PRINT_INPUT_ERROR("Factorized hopping requires rc_hopping <= rc_radial.");
  }
  if (n_max_radial < 0 || n_max_angular < 0 || basis_size_radial < 0 || basis_size_angular < 0) {
    PRINT_INPUT_ERROR("Descriptor orders must be non-negative.");
  }
  if (
    n_max_radial > 16 || n_max_angular > 16 || basis_size_radial > 16 || basis_size_angular > 16) {
    PRINT_INPUT_ERROR(
      "n_max and basis_size cannot exceed 16 in the current NEP descriptor implementation.");
  }
  if (L_max <= 0 || L_max > 8) {
    PRINT_INPUT_ERROR("l_max must be in [1, 8].");
  }
  if (num_neurons_onsite <= 0 || num_neurons_hopping <= 0) {
    PRINT_INPUT_ERROR("Neuron counts must be positive.");
  }
  if (maximum_epoch <= 0) {
    PRINT_INPUT_ERROR("epoch must be positive.");
  }
  if (seed < 0) {
    PRINT_INPUT_ERROR("seed must be non-negative.");
  }
  if (learning_rate <= 0.0 || !std::isfinite(learning_rate)) {
    PRINT_INPUT_ERROR("learning_rate must be finite and positive.");
  }
  if (weight_decay < 0.0 || !std::isfinite(weight_decay)) {
    PRINT_INPUT_ERROR("weight_decay must be finite and non-negative.");
  }
  if (!(lr_factor > 0.0 && lr_factor < 1.0) || !std::isfinite(lr_factor)) {
    PRINT_INPUT_ERROR("lr_scheduler factor must lie in (0,1).");
  }
  if (lr_patience < 0) {
    PRINT_INPUT_ERROR("lr_scheduler patience must be non-negative.");
  }
  if (lr_rel_threshold < 0.0 || !std::isfinite(lr_rel_threshold)) {
    PRINT_INPUT_ERROR("lr_scheduler relative threshold must be finite and non-negative.");
  }
  if (lr_min <= 0.0 || !std::isfinite(lr_min) || lr_min > learning_rate) {
    PRINT_INPUT_ERROR("lr_scheduler min_lr must be positive and <= learning_rate.");
  }
  if (structure_batch <= 0 || k_batch <= 0) {
    PRINT_INPUT_ERROR("Both batch sizes must be positive.");
  }
  if (band_min < 0 || band_max <= band_min) {
    PRINT_INPUT_ERROR("band_range must satisfy 0 <= min < max.");
  }
  if (band_curriculum < 0) {
    PRINT_INPUT_ERROR("band_curriculum must be non-negative.");
  }
  for (const auto& range : band_weight_ranges_) {
    if (
      range.band_min < band_min || range.band_max > band_max || range.band_max <= range.band_min) {
      PRINT_INPUT_ERROR("Each band_weight range must lie inside band_range and satisfy min < max.");
    }
    if (!std::isfinite(range.weight) || range.weight < 0.0) {
      PRINT_INPUT_ERROR("band_weight values must be finite and non-negative.");
    }
  }
  if (!std::isfinite(dispersion_weight) || dispersion_weight < 0.0) {
    PRINT_INPUT_ERROR("dispersion weight must be finite and non-negative.");
  }
  if (!(dispersion_sigma > 0.0) || std::isnan(dispersion_sigma)) {
    PRINT_INPUT_ERROR("dispersion sigma must be positive or 'inf'.");
  }
}

int Parameters::hopping_block_size(const int outputs) const
{
  return num_neurons_hopping * dim + num_neurons_hopping + dim_radial * num_neurons_hopping +
         dim_radial + outputs * dim_radial + outputs;
}

void Parameters::derive_model_sizes()
{
  if (basis == "sp3") {
    orbitals_per_atom = 4;
    num_onsite_outputs = 2;
    num_hopping_outputs_homo = 4;
    num_hopping_outputs_hetero = 5;
  } else if (basis == "sp3sstar") {
    orbitals_per_atom = 5;
    num_onsite_outputs = 3;
    num_hopping_outputs_homo = 7;
    num_hopping_outputs_hetero = 10;
  } else if (basis == "sp3d5") {
    orbitals_per_atom = 9;
    num_onsite_outputs = 3;
    num_hopping_outputs_homo = 10;
    num_hopping_outputs_hetero = 14;
  } else if (basis == "sp3d5sstar") {
    orbitals_per_atom = 10;
    num_onsite_outputs = 4;
    num_hopping_outputs_homo = 14;
    num_hopping_outputs_hetero = 21;
  } else {
    PRINT_INPUT_ERROR("Internal error: unsupported basis.");
  }
  max_hopping_outputs = (num_types > 1) ? num_hopping_outputs_hetero : num_hopping_outputs_homo;

  dim_radial = n_max_radial + 1;
  dim_angular = (n_max_angular + 1) * L_max;
  dim = dim_radial + dim_angular;
  if (dim_angular > 90) {
    PRINT_INPUT_ERROR("The angular descriptor dimension cannot exceed 90.");
  }

  num_c_radial = (n_max_radial + 1) * (basis_size_radial + 1);
  num_c_angular = (n_max_angular + 1) * (basis_size_angular + 1);
  descriptor_block_size = num_c_radial + num_c_angular;

  onsite_block_size = num_neurons_onsite * dim + num_neurons_onsite +
                      num_onsite_outputs * num_neurons_onsite + num_onsite_outputs;
  number_of_variables_onsite = num_types * onsite_block_size;

  num_element_pairs = num_types * (num_types + 1) / 2;
  hopping_pair_offsets.assign(num_element_pairs, 0);
  hopping_pair_outputs.assign(num_element_pairs, 0);
  number_of_variables_hopping = 0;
  for (int a = 0; a < num_types; ++a) {
    for (int b = a; b < num_types; ++b) {
      const int p = unordered_type_pair_index(a, b);
      const int outputs = hopping_outputs_for_pair(a, b);
      hopping_pair_offsets[p] = number_of_variables_hopping;
      hopping_pair_outputs[p] = outputs;
      number_of_variables_hopping += hopping_block_size(outputs);
    }
  }

  number_of_variables_descriptor = num_types * num_types * descriptor_block_size;

  offset_onsite = 0;
  offset_hopping = number_of_variables_onsite;
  offset_descriptor = offset_hopping + number_of_variables_hopping;
  number_of_variables = offset_descriptor + number_of_variables_descriptor;
}

void Parameters::derive_band_weights()
{
  const int num_selected = band_max - band_min;
  band_weights.assign(num_selected, 1.0);

  for (const auto& range : band_weight_ranges_) {
    for (int n = range.band_min; n < range.band_max; ++n) {
      band_weights[n - band_min] = range.weight;
    }
  }

  band_weight_sum = 0.0;
  for (const double weight : band_weights) {
    band_weight_sum += weight;
  }
  if (!(band_weight_sum > 0.0)) {
    PRINT_INPUT_ERROR("At least one selected band must have a positive band_weight.");
  }
  if (band_curriculum > 0 && !(band_weights.front() > 0.0)) {
    PRINT_INPUT_ERROR(
      "band_curriculum requires the lowest selected band to have positive band_weight.");
  }
}

int Parameters::unordered_type_pair_index(int type_a, int type_b) const
{
  if (type_a < 0 || type_a >= num_types || type_b < 0 || type_b >= num_types) {
    return -1;
  }
  if (type_a > type_b) {
    std::swap(type_a, type_b);
  }
  return type_a * num_types - type_a * (type_a - 1) / 2 + (type_b - type_a);
}

int Parameters::hopping_outputs_for_pair(const int type_a, const int type_b) const
{
  return type_a == type_b ? num_hopping_outputs_homo : num_hopping_outputs_hetero;
}

void Parameters::print() const
{
  print_line_1();
  printf("TB parameters\n");
  print_line_2();
  printf("  type                   = %d", num_types);
  for (const auto& element : elements) {
    printf(" %s", element.c_str());
  }
  printf("\n");
  printf("  orbital basis          = %s\n", basis.c_str());
  printf("  orbitals per atom      = %d\n", orbitals_per_atom);
  printf("  onsite outputs/type    = %d\n", num_onsite_outputs);
  printf(
    "  hopping outputs        = homo %d, hetero %d\n",
    num_hopping_outputs_homo,
    num_hopping_outputs_hetero);
  printf("  cutoff radial/angular  = %g / %g Angstrom\n", rc_radial, rc_angular);
  printf("  cutoff hopping         = %g Angstrom\n", rc_hopping);
  printf("  n_max radial/angular   = %d / %d\n", n_max_radial, n_max_angular);
  printf("  basis radial/angular   = %d / %d\n", basis_size_radial, basis_size_angular);
  printf("  l_max                  = %d\n", L_max);
  printf("  descriptor dimension   = %d (radial %d, angular %d)\n", dim, dim_radial, dim_angular);
  printf("  onsite ANN             = %d-%d-%d\n", dim, num_neurons_onsite, num_onsite_outputs);
  printf("  hopping Env ANN        = %d-%d-%d\n", dim, num_neurons_hopping, dim_radial);
  printf("  hopping output         = %d-%d (linear)\n", dim_radial, max_hopping_outputs);
  printf("  onsite ANN blocks      = %d\n", num_types);
  printf("  hopping ANN blocks     = %d unordered element pairs\n", num_element_pairs);
  printf("  descriptor c_nk blocks = %d ordered center-neighbor pairs\n", num_types * num_types);
  printf("  trainable parameters   = %d\n", number_of_variables);
  printf("    onsite               = %d\n", number_of_variables_onsite);
  printf("    hopping              = %d\n", number_of_variables_hopping);
  printf("    descriptor           = %d\n", number_of_variables_descriptor);
  if (!prediction) {
    printf("  optimizer              = AdamW (analytic gradient)\n");
    printf("  training epochs        = %d\n", maximum_epoch);
    printf("  learning rate          = %g\n", learning_rate);
    printf("  decoupled weight decay = %g\n", weight_decay);
    printf("  lr scheduler           = plateau factor=%g patience=", lr_factor);
    if (lr_scheduler_explicit) {
      printf("%d", lr_patience);
    } else {
      printf("auto");
    }
    printf(" rel_threshold=%g min_lr=%g\n", lr_rel_threshold, lr_min);
    printf("  optimizer seed         = %d\n", seed);
    printf(
      "  batch structures/k     = %d / %d (each capped by data size)\n", structure_batch, k_batch);
  }
  printf("  band range             = [%d, %d)\n", band_min, band_max);
  if (prediction) {
    printf("  calculation mode       = prediction\n");
  } else if (band_curriculum > 0) {
    printf(
      "  band curriculum        = %d epoch%s per added band\n",
      band_curriculum,
      band_curriculum == 1 ? "" : "s");
  } else {
    printf("  band curriculum        = off\n");
  }
  printf("  band weights           =");
  for (int n = 0; n < static_cast<int>(band_weights.size()); ++n) {
    printf(" %d:%g", band_min + n, band_weights[n]);
  }
  printf("\n");
  printf("  dispersion weight      = %g 1/Angstrom\n", dispersion_weight);
  if (std::isinf(dispersion_sigma)) {
    printf("  dispersion sigma       = inf\n");
  } else {
    printf("  dispersion sigma       = %g 1/Angstrom\n", dispersion_sigma);
  }
}
