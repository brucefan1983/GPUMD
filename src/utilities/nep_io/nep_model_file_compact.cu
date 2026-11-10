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

#include "nep_model_file.cuh"
#include <algorithm>

static int triangular_pair_index(int type1, int type2, int num_types)
{
  if (type1 > type2) {
    int temp = type1;
    type1 = type2;
    type2 = temp;
  }
  return type1 * num_types - type1 * (type1 - 1) / 2 + type2 - type1;
}

bool NEP_Model_File::compact(
  const std::vector<std::string>& active_species,
  NEP_File& document,
  std::string& error) const
{
  error.clear();
  std::vector<int> active_to_full;
  for (const std::string& element : active_species) {
    const auto found = std::find(elements.begin(), elements.end(), element);
    if (found == elements.end()) {
      error = "The active species is not supported by the potential.";
      return false;
    }
    const int type = static_cast<int>(found - elements.begin());
    if (std::find(active_to_full.begin(), active_to_full.end(), type) != active_to_full.end()) {
      return fail(error, "duplicate active element.");
    }
    active_to_full.push_back(type);
  }
  if (active_to_full.empty()) {
    return fail(error, "at least one active element is required.");
  }
  compact_parameters(document, active_to_full);
  document.elements = active_species;
  return true;
}

void NEP_Model_File::compact_parameters(
  NEP_File& document, const std::vector<int>& active_to_full) const
{
  for (NEP_File::Field& field : document.fields) {
    if (field.name == "energy_reference") {
      std::vector<std::string> values;
      for (int t : active_to_full) {
        values.push_back(field.values[t]);
      }
      field.values.swap(values);
    } else if (field.name == "cutoff" && has_multiple_cutoffs) {
      std::vector<std::string> values;
      for (int t : active_to_full) {
        values.push_back(field.values[2 * t]);
        values.push_back(field.values[2 * t + 1]);
      }
      values.insert(values.end(), field.values.end() - 2, field.values.end());
      field.values.swap(values);
    }
  }
  for (NEP_File::Block& block : document.blocks) {
    std::vector<double> values;
    if (block.name == "parameters") {
      for (int ann = 0; ann < layout.ann_count; ++ann) {
        const int start = ann * layout.number_of_variables_ann;
        for (int t : active_to_full) {
          const int offset = start + t * layout.number_of_variables_ann_1;
          values.insert(
            values.end(),
            block.values.begin() + offset,
            block.values.begin() + offset + layout.number_of_variables_ann_1);
        }
        const int globals = start + num_types * layout.number_of_variables_ann_1;
        values.insert(
          values.end(),
          block.values.begin() + globals,
          block.values.begin() + globals + layout.global_parameters);
      }
      int start = layout.ann_count * layout.number_of_variables_ann;
      const int pairs = num_types * num_types;
      const int basis_counts[] = {
        (n_max_radial + 1) * (basis_size_radial + 1),
        (n_max_angular + 1) * (basis_size_angular + 1)};
      for (const int basis_count : basis_counts) {
        for (int t1 : active_to_full) {
          for (int t2 : active_to_full) {
            const int offset = start + (t1 * num_types + t2) * basis_count;
            values.insert(
              values.end(), block.values.begin() + offset,
              block.values.begin() + offset + basis_count);
          }
        }
        start += pairs * basis_count;
      }
      block.values.swap(values);
    } else if (block.name == "zbl_parameters") {
      for (int t1 = 0; t1 < static_cast<int>(active_to_full.size()); ++t1) {
        for (int t2 = t1; t2 < static_cast<int>(active_to_full.size()); ++t2) {
          const int offset =
            10 * triangular_pair_index(active_to_full[t1], active_to_full[t2], num_types);
          values.insert(
            values.end(), block.values.begin() + offset, block.values.begin() + offset + 10);
        }
      }
      block.values.swap(values);
    }
  }
}
