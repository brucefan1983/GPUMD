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

#include "nep_charge_reader.cuh"
#include "nep_charge_parameters.cuh"
#include "main_nep/xyz_reader.cuh"
#include "utilities/error.cuh"

void NEP_Charge_Reader::read_structure(const XYZ_Record& record, Structure& structure)
{
  read_nep_boundaries(record, structure);
  NEP_Energy_Reader::read_structure(record, structure);
  labels.emplace_back();
  auto& label = labels.back();
  const auto& tokens = record.tokens;
  const auto& xyz_filename = record.filename;
  int line_number = record.comment_line_number;
  // get total charge (optional; default is 0)
  for (const auto& token : tokens) {
    const std::string charge_string = "charge=";
    if (token.substr(0, charge_string.length()) == charge_string) {
      label.charge = get_double_from_token(
        token.substr(charge_string.length(), token.length()), xyz_filename.c_str(), line_number);
    }
  }

  int bec_offset = record.get_property_offset("bec");
  label.has_bec = bec_offset >= 0;
  label.bec.resize(structure.num_atom * 9);
  if (label.has_bec) {
    charge_parameters.has_bec = true;
    for (int na = 0; na < structure.num_atom; ++na) {
      for (int d = 0; d < 9; ++d) {
        label.bec[na * 9 + d] = get_double_from_token(
          record.atoms[na][d + bec_offset], xyz_filename.c_str(), line_number + na + 1);
      }
    }
  }
}

void NEP_Charge_Reader::print_summary(const std::vector<Structure>& structures) const
{
  print_nep_boundaries(structures);
  NEP_Energy_Reader::print_summary(structures);
}

const char* NEP_Charge_Reader::energy_difference_conflict(
  const Structure& a, const Structure& b) const
{
  const char* conflict = NEP_Energy_Reader::energy_difference_conflict(a, b);
  if (conflict && labels[a.index_in_file].charge != labels[b.index_in_file].charge) {
    return "differ only in charge=, for which a qNEP model predicts no meaningful energy difference.";
  }
  return conflict;
}

void NEP_Charge_Reader::prepare_dataset(Dataset& dataset)
{
  NEP_Energy_Reader::prepare_dataset(dataset);
  charge_data.create(dataset).construct(dataset, labels);
}
