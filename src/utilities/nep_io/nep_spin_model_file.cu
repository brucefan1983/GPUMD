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
  Chengbing Chen: Initial implementation.
------------------------------------------------------------------------------*/

#include "nep_file.cuh"
#include "nep_spin_model_file.cuh"
#include "utilities/nep_spin/spin_layout.cuh"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace
{
bool fail(std::string& error, const std::string& message)
{
  error = "NEP-spin model: " + message;
  return false;
}

bool to_float(double value, float& result)
{
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
    return false;
  result = static_cast<float>(value);
  return value == 0 || result != 0;
}
} // namespace

bool parse_spin_wall(
  const std::vector<std::string>& values,
  const std::vector<int>& active,
  std::vector<Spin_Walls>& walls,
  std::vector<Spin_Walls>* automatic,
  std::string& error)
{
  walls.clear();
  if (automatic)
    automatic->clear();
  for (size_t i = 0; i < values.size();) {
    Spin_Walls numeric{}, margins{};
    for (int side = 0; side < Spin_Walls::count; ++side) {
      const bool is_auto = i < values.size() && values[i] == "auto";
      if (is_auto) {
        if (!automatic)
          return fail(error, "spin_wall auto is only allowed in training input.");
        ++i;
      }
      if (values.size() - i < 2)
        return fail(error, "spin_wall requires a lower and an upper wall in each group.");
      Spin_Wall wall;
      if (
        !NEP_File::parse_double(values[i], wall.threshold) ||
        !NEP_File::parse_double(values[i + 1], wall.stiffness) || wall.threshold < 0 ||
        wall.stiffness < 0)
        return fail(
          error, "spin_wall requires finite non-negative bounds, margins, and stiffnesses.");
      i += 2;
      if (wall.stiffness == 0)
        continue;
      if (!is_auto && wall.threshold == 0)
        return fail(error, "spin_wall requires a positive threshold when stiffness is nonzero.");
      (is_auto ? margins : numeric).side[side] = wall;
    }
    walls.push_back(numeric);
    if (automatic)
      automatic->push_back(margins);
  }
  const bool shared = walls.size() == 1;
  if (!shared && walls.size() != active.size())
    return fail(error, "spin_wall requires one shared group or one group per element.");
  if (shared) {
    walls.resize(active.size(), walls.front());
    if (automatic)
      automatic->resize(active.size(), automatic->front());
  }
  for (size_t t = 0; t < active.size(); ++t) {
    if (active[t])
      continue;
    if (!shared && (walls[t].enabled() || (automatic && (*automatic)[t].enabled())))
      return fail(error, "spin_wall can only be enabled for active spin types.");
    walls[t] = {};
    if (automatic)
      (*automatic)[t] = {};
  }
  return true;
}

bool NEP_Spin_Model_File::read(std::istream& input, std::string& error)
{
  *this = NEP_Spin_Model_File{};
  NEP_File doc;
  if (!doc.read(input, error))
    return false;
  if (doc.find_field("model")->values[0] != "nep_spin")
    return fail(error, "expected model nep_spin.");
  elements = doc.elements;
  const int types = elements.size();
  std::vector<std::string> handled = {"format_version", "model"};
  auto field = [&](const char* name, size_t lo, size_t hi) -> const NEP_File::Field* {
    const auto* f = doc.find_field(name);
    if (!f || f->values.size() < lo || f->values.size() > hi) {
      fail(error, std::string("missing or malformed field '") + name + "'.");
      return nullptr;
    }
    handled.push_back(name);
    return f;
  };
  auto integer = [&](const NEP_File::Field* f, size_t i, int lo, int hi, int& result) {
    if (!f)
      return false;
    size_t value;
    if (!NEP_File::parse_size(f->values[i], value) || value < size_t(lo) || value > size_t(hi))
      return fail(error, "invalid integer in '" + f->name + "'.");
    result = static_cast<int>(value);
    return true;
  };
  auto number = [&](const NEP_File::Field* f, size_t i, float& result) {
    if (!f)
      return false;
    double value;
    if (!NEP_File::parse_double(f->values[i], value) || !to_float(value, result))
      return fail(error, "invalid float in '" + f->name + "'.");
    return true;
  };
  auto scalar = [&](const char* name, int lo, int hi, int& result) {
    return integer(field(name, 1, 1), 0, lo, hi, result);
  };
  const auto* f = field("spin_version", 1, 1);
  if (!f)
    return false;
  int spin_version;
  if (!integer(f, 0, 1, 1, spin_version))
    return false;

  f = field("cutoff", 4, 4);
  if (
    !number(f, 0, rc_radial) || !number(f, 1, rc_angular) ||
    !integer(f, 2, 0, std::numeric_limits<int>::max(), max_radial) ||
    !integer(f, 3, 0, std::numeric_limits<int>::max(), max_angular))
    return false;
  if (rc_angular < 3 || rc_angular > rc_radial || rc_radial > 100)
    return fail(error, "cutoff requires 3 <= angular <= radial <= 100.");
  f = field("n_max", 2, 2);
  if (!integer(f, 0, 0, 12, n_max[0]) || !integer(f, 1, 0, 8, n_max[1]))
    return false;
  f = field("basis_size", 2, 2);
  if (!integer(f, 0, 0, 16, basis_size[0]) || !integer(f, 1, 0, 12, basis_size[1]))
    return false;
  f = field("l_max", 3, 7);
  if (!integer(f, 0, 2, 8, l_max[0]))
    return false;
  for (size_t i = 1; i < f->values.size(); ++i)
    if (!integer(f, i, 0, i == 1 ? 2 : 1, l_max[i]))
      return false;
  if (l_max[1] == 1)
    return fail(error, "l_max q_222 must be 0 or 2.");
  if (const char* message = angular_rank_error(l_max[0], l_max[4], l_max[5], l_max[6]))
    return fail(error, message);
  f = field("ann", 2, 2);
  int second_layer;
  if (!integer(f, 0, 1, 120, num_neurons) || !integer(f, 1, 0, 0, second_layer))
    return false;

  int scaler_flag;
  if (
    !scalar("spin_basis_size", basis_size[1], basis_size[1], spin_basis_size) ||
    !scalar("spin_l_max", 0, 4, spin_l_max) || !scalar("spin_compress", 1, 9, spin_compress) ||
    !scalar("spin_order", 1, 4, spin_order) || !scalar("spin_soc", 0, 1, spin_soc) ||
    !scalar("spin_scaler", 1, 1, scaler_flag))
    return false;
  projection_parameter_count =
    nep_spin::Layout(spin_compress, spin_l_max, spin_order, spin_soc).parameter_count(n_max[1] + 1);
  f = field("spin_cutoff", 1, types);
  if (!f)
    return false;
  if (f->values.size() != 1 && f->values.size() != size_t(types))
    return fail(error, "spin_cutoff requires one value or one per element.");
  spin_cutoff.resize(types);
  for (int t = 0; t < types; ++t) {
    if (!number(f, f->values.size() == 1 ? 0 : t, spin_cutoff[t]))
      return false;
    if (spin_cutoff[t] < 3)
      return fail(error, "spin_cutoff must be at least 3 A.");
    if (spin_cutoff[t] > rc_radial)
      return fail(error, "spin_cutoff must not exceed the NEP radial cutoff.");
  }
  auto active_types = [&](const char* name, std::vector<int>& active) {
    const auto* selection = field(name, 1, types);
    if (!selection)
      return false;
    active.assign(types, 0);
    for (const auto& symbol : selection->values) {
      auto found = std::find(elements.begin(), elements.end(), symbol);
      if (found == elements.end())
        return fail(error, std::string("unknown element in ") + name);
      const int t = found - elements.begin();
      if (active[t])
        return fail(error, std::string("duplicate element in ") + name);
      active[t] = 1;
    }
    return true;
  };
  spin_type.assign(types, 1);
  if (doc.find_field("spin_type") && !active_types("spin_type", spin_type))
    return false;

  spin_walls.resize(types);
  if (doc.find_field("spin_wall")) {
    f = field("spin_wall", 4 * types, 4 * types);
    if (!f || !parse_spin_wall(f->values, spin_type, spin_walls, nullptr, error))
      return false;
    for (const auto& walls : spin_walls)
      if (!walls.ordered())
        return fail(error, "spin_wall lower threshold must be below the upper threshold.");
  }

  f = field("energy_reference", types, types);
  if (!f)
    return false;
  energy_reference.resize(types);
  for (int t = 0; t < types; ++t)
    if (!NEP_File::parse_double(f->values[t], energy_reference[t]))
      return fail(error, "invalid energy_reference.");
  enable_zbl = doc.find_field("zbl") != nullptr;
  if (enable_zbl) {
    f = field("zbl", 2, 3);
    if (!number(f, 0, zbl_rc_inner) || !number(f, 1, zbl_rc_outer))
      return false;
    if (zbl_rc_inner == 0 && zbl_rc_outer == 0)
      return fail(error, "NEP-spin does not support flexible ZBL (zbl 0 0).");
    if (zbl_rc_inner < 0 || zbl_rc_outer <= zbl_rc_inner)
      return fail(error, "zbl requires 0 <= inner < outer.");
    use_typewise_cutoff_zbl = f->values.size() == 3;
    if (use_typewise_cutoff_zbl) {
      if (!number(f, 2, typewise_cutoff_zbl_factor))
        return false;
      if (typewise_cutoff_zbl_factor < 0.5f)
        return fail(error, "typewise ZBL cutoff factor must be at least 0.5.");
    }
  }
  int angular_channels = l_max[0] + (l_max[1] != 0);
  for (int i = 2; i < 7; ++i)
    angular_channels += l_max[i];
  const int dim_angular = (n_max[1] + 1) * angular_channels;
  dim_struct = n_max[0] + 1 + dim_angular;
  dim_spin = nep_spin::Layout(spin_compress, spin_l_max, spin_order, spin_soc).dimension;
  dim = dim_struct + dim_spin;
  if (dim_angular > 90 || dim > MAX_DIM_SPIN)
    return fail(error, "descriptor dimensions exceed the supported Spin shape.");
  const int parameter_count =
    (dim + 2) * num_neurons * types + 1 +
    types * types * ((n_max[0] + 1) * (basis_size[0] + 1) + (n_max[1] + 1) * (basis_size[1] + 1)) +
    projection_parameter_count;
  auto block = [&](const char* name, int count, std::vector<float>& values, bool positive) {
    const auto* b = doc.find_block(name);
    if (!b || b->values.size() != size_t(count))
      return fail(error, std::string("incorrect count in block '") + name + "'.");
    values.resize(count);
    for (int i = 0; i < count; ++i) {
      if (!to_float(b->values[i], values[i]) || (positive && values[i] <= 0))
        return fail(error, std::string("invalid value in block '") + name + "'.");
    }
    return true;
  };
  return block("parameters", parameter_count, parameters, false) &&
         block("q_scaler", dim, q_scaler, true) &&
         doc.check_unhandled(handled, {"parameters", "q_scaler"}, error);
}

bool NEP_Spin_Model_File::is_spin_model(const std::string& filename)
{
  std::ifstream input(filename);
  std::string model;
  std::string error;
  return NEP_File::read_model_name(input, model, error) && model == "nep_spin";
}
