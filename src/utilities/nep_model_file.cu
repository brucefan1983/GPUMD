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

#include "nep_file.cuh"
#include "nep_model_file.cuh"
#include <cmath>
#include <limits>
#include <utility>

bool NEP_Model_File::fail(std::string& error, const std::string& message, std::size_t line)
{
  error = "NEP model" + (line ? " line " + std::to_string(line) : "") + ": " + message;
  return false;
}

bool NEP_Model_File::read_integer(
  const NEP_File::Field& field,
  std::size_t index,
  int minimum,
  int maximum,
  int& value,
  std::string& error)
{
  std::size_t parsed;
  if (
    !NEP_File::parse_size(field.values[index], parsed) ||
    parsed < static_cast<std::size_t>(minimum) ||
    parsed > static_cast<std::size_t>(maximum)) {
    return fail(
      error,
      "'" + field.name + "' requires an integer in [" + std::to_string(minimum) + ", " +
        std::to_string(maximum) + "].",
      field.line);
  }
  value = static_cast<int>(parsed);
  return true;
}

static bool to_float(double value, float& result)
{
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) {
    return false;
  }
  result = static_cast<float>(value);
  return value == 0.0 || result != 0.0f;
}

bool NEP_Model_File::read_float(
  const NEP_File::Field& field, std::size_t index, float& value, std::string& error)
{
  double parsed;
  if (!NEP_File::parse_double(field.values[index], parsed) || !to_float(parsed, value)) {
    return fail(error, "invalid float value in '" + field.name + "'.", field.line);
  }
  return true;
}

const NEP_File::Field* NEP_Model_File::require_field(
  const NEP_File& document,
  const char* name,
  std::size_t minimum,
  std::size_t maximum,
  std::vector<std::string>& handled,
  std::string& error)
{
  const NEP_File::Field* field = document.find_field(name);
  if (field == nullptr) {
    fail(error, "missing field '" + std::string(name) + "'.");
    return nullptr;
  }
  if (field->values.size() < minimum || field->values.size() > maximum) {
    fail(error, "incorrect number of values in '" + field->name + "'.", field->line);
    return nullptr;
  }
  handled.push_back(name);
  return field;
}

bool NEP_Model_File::read_block(
  const NEP_File& document,
  const char* name,
  int count,
  std::vector<float>& values,
  std::vector<std::string>& handled,
  std::string& error)
{
  const NEP_File::Block* block = document.find_block(name);
  if (block == nullptr) {
    return fail(error, "missing block '" + std::string(name) + "'.");
  }
  if (block->values.size() != static_cast<std::size_t>(count)) {
    return fail(
      error,
      "block '" + block->name + "' requires " + std::to_string(count) + " values, found " +
        std::to_string(block->values.size()) + ".",
      block->line);
  }
  values.resize(count);
  for (int i = 0; i < count; ++i) {
    if (!to_float(block->values[i], values[i])) {
      return fail(error, "invalid float value in block '" + block->name + "'.", block->line);
    }
  }
  handled.push_back(name);
  return true;
}

bool NEP_Model_File::read_descriptor(
  const NEP_File& document, std::vector<std::string>& handled, std::string& error)
{
  const NEP_File::Field* field =
    require_field(document, "cutoff", 4, 2 * num_types + 2, handled, error);
  if (field == nullptr) {
    return false;
  }
  has_multiple_cutoffs = field->values.size() != 4;
  const int count = has_multiple_cutoffs ? num_types : 1;
  if (field->values.size() != static_cast<std::size_t>(2 * count + 2)) {
    return fail(
      error,
      "cutoff requires two radii or one pair per element, followed by two neighbor counts.",
      field->line);
  }
  rc_radial.resize(num_types);
  rc_angular.resize(num_types);
  for (int t = 0; t < num_types; ++t) {
    const int offset = has_multiple_cutoffs ? 2 * t : 0;
    if (
      !read_float(*field, offset, rc_radial[t], error) ||
      !read_float(*field, offset + 1, rc_angular[t], error)) {
      return false;
    }
    if (rc_angular[t] < 3.0f || rc_angular[t] > rc_radial[t] || rc_radial[t] > 100.0f) {
      return fail(error, "cutoff requires 3 <= angular <= radial <= 100 Angstrom.", field->line);
    }
  }
  const int max_count = std::numeric_limits<int>::max();
  if (
    !read_integer(*field, 2 * count, 0, max_count, max_radial, error) ||
    !read_integer(*field, 2 * count + 1, 0, max_count, max_angular, error)) {
    return false;
  }
  field = require_field(document, "n_max", 2, 2, handled, error);
  if (
    field == nullptr || !read_integer(*field, 0, 0, 15, n_max_radial, error) ||
    !read_integer(*field, 1, 0, 12, n_max_angular, error)) {
    return false;
  }
  field = require_field(document, "basis_size", 2, 2, handled, error);
  if (
    field == nullptr || !read_integer(*field, 0, 0, 16, basis_size_radial, error) ||
    !read_integer(*field, 1, 0, 12, basis_size_angular, error)) {
    return false;
  }
  field = require_field(document, "l_max", 3, 7, handled, error);
  if (field == nullptr || !read_integer(*field, 0, 2, 8, L_max, error)) {
    return false;
  }
  int flags[6] = {0, 0, 0, 0, 0, 0};
  for (std::size_t i = 1; i < field->values.size(); ++i) {
    if (!read_integer(*field, i, 0, i == 1 ? 2 : 1, flags[i - 1], error)) {
      return false;
    }
  }
  if (flags[0] == 1) {
    return fail(error, "the q_222 value in l_max must be 0 or 2.", field->line);
  }
  has_q_222 = flags[0] != 0;
  has_q_1111 = flags[1];
  has_q_112 = flags[2];
  has_q_123 = flags[3];
  has_q_233 = flags[4];
  has_q_134 = flags[5];
  layout.dim_radial = n_max_radial + 1;
  layout.dim_angular = (n_max_angular + 1) *
                       (L_max + has_q_222 + has_q_1111 + has_q_112 + has_q_123 + has_q_233 +
                        has_q_134);
  if (layout.dim_angular > 90) {
    return fail(error, "number of angular descriptors exceeds 90.", field->line);
  }
  layout.dim = layout.dim_radial + layout.dim_angular;
  return true;
}

bool NEP_Model_File::read_ann(
  const NEP_File& document, std::vector<std::string>& handled, std::string& error)
{
  const NEP_File::Field* field = require_field(document, "ann", 2, 2, handled, error);
  if (
    field == nullptr || !read_integer(*field, 0, 1, 120, num_neurons1, error) ||
    !read_integer(*field, 1, 0, 120, num_neurons2, error)) {
    return false;
  }
  return true;
}

bool NEP_Model_File::read_reference(
  const NEP_File& document, std::vector<std::string>& handled, std::string& error)
{
  const NEP_File::Field* field =
    require_field(document, "energy_reference", num_types, num_types, handled, error);
  if (field == nullptr) {
    return false;
  }
  energy_reference.resize(num_types);
  for (int t = 0; t < num_types; ++t) {
    if (!NEP_File::parse_double(field->values[t], energy_reference[t])) {
      return fail(error, "invalid double value in energy_reference.", field->line);
    }
  }
  return true;
}

bool NEP_Model_File::read_zbl(
  const NEP_File& document,
  std::vector<std::string>& fields,
  std::vector<std::string>& blocks,
  std::string& error)
{
  if (document.find_field("zbl") == nullptr) {
    return true;
  }
  const NEP_File::Field* field = require_field(document, "zbl", 2, 3, fields, error);
  if (
    field == nullptr || !read_float(*field, 0, zbl_rc_inner, error) ||
    !read_float(*field, 1, zbl_rc_outer, error)) {
    return false;
  }
  enable_zbl = true;
  flexible_zbl = zbl_rc_inner == 0.0f && zbl_rc_outer == 0.0f;
  if (flexible_zbl) {
    if (field->values.size() != 2) {
      return fail(error, "flexible ZBL cannot use a typewise cutoff factor.", field->line);
    }
    const int count = 10 * num_types * (num_types + 1) / 2;
    if (!read_block(document, "zbl_parameters", count, zbl_parameters, blocks, error)) {
      return false;
    }
    for (int i = 0; i < count; i += 10) {
      if (zbl_parameters[i] < 0.0f || zbl_parameters[i + 1] <= zbl_parameters[i]) {
        return fail(
          error,
          "each zbl_parameters pair requires 0 <= inner < outer.",
          document.find_block("zbl_parameters")->line);
      }
    }
  } else {
    if (zbl_rc_inner < 0.0f || zbl_rc_outer <= zbl_rc_inner) {
      return fail(error, "zbl requires 0 <= inner < outer, or 0 0 for flexible ZBL.", field->line);
    }
    if (field->values.size() == 3) {
      use_typewise_cutoff_zbl = true;
      if (!read_float(*field, 2, typewise_cutoff_zbl_factor, error)) {
        return false;
      }
      if (typewise_cutoff_zbl_factor < 0.5f) {
        return fail(error, "typewise ZBL cutoff factor must be at least 0.5.", field->line);
      }
    }
  }
  return true;
}

void NEP_Model_File::calculate_layout(
  int extra_inputs, int extra_outputs, int globals, int ann_count)
{
  layout.dim += extra_inputs;
  layout.ann_count = ann_count;
  layout.global_parameters = globals;
  if (num_neurons2 > 0) {
    layout.number_of_variables_ann_1 =
      (layout.dim + 1) * num_neurons1 + (num_neurons1 + 2) * num_neurons2;
  } else {
    layout.number_of_variables_ann_1 = (layout.dim + 2 + extra_outputs) * num_neurons1;
  }
  layout.number_of_variables_ann = layout.number_of_variables_ann_1 * num_types + globals;
  layout.number_of_variables_descriptor =
    num_types * num_types *
    ((n_max_radial + 1) * (basis_size_radial + 1) +
     (n_max_angular + 1) * (basis_size_angular + 1));
  layout.number_of_variables =
    layout.number_of_variables_ann * ann_count + layout.number_of_variables_descriptor;
}

bool NEP_Model_File::read_parameters(
  const NEP_File& document, std::vector<std::string>& blocks, std::string& error)
{
  if (layout.dim > 103) {
    return fail(error, "descriptor dimension exceeds 103.", document.find_field("l_max")->line);
  }
  return read_block(document, "parameters", layout.number_of_variables, parameters, blocks, error) &&
         read_block(document, "q_scaler", layout.dim, q_scaler, blocks, error);
}

bool NEP_Model_File::read(std::istream& input, std::string& error)
{
  *this = NEP_Model_File();
  NEP_File document;
  if (!document.read(input, error)) {
    return false;
  }
  return parse(document, error);
}

bool NEP_Model_File::parse(const NEP_File& document, std::string& error)
{
  NEP_Model_File parsed;
  parsed.name = document.find_field("model")->values[0];
  parsed.elements = document.elements;
  parsed.num_types = static_cast<int>(parsed.elements.size());
  std::vector<std::string> fields = {"format_version", "model"};
  std::vector<std::string> blocks;
  bool success = false;
  if (parsed.name == "nep" || parsed.name == "nep_temperature") {
    success = parsed.read_nep(document, fields, blocks, error);
  } else if (parsed.name == "qnep1" || parsed.name == "qnep2") {
    success = parsed.read_charge(document, fields, blocks, error);
  } else if (parsed.name == "nep_dipole" || parsed.name == "nep_polarizability") {
    success = parsed.read_tnep(document, fields, blocks, error);
  } else if (parsed.name == "nep_vdw") {
    success = parsed.read_vdw(document, fields, blocks, error);
  } else if (parsed.name == "qnep3_vdw") {
    success = parsed.read_charge_vdw(document, fields, blocks, error);
  } else {
    return fail(
      error, "unsupported model '" + parsed.name + "'.", document.find_field("model")->line);
  }
  if (!success || !document.check_unhandled(fields, blocks, error)) {
    return false;
  }
  *this = std::move(parsed);
  return true;
}
