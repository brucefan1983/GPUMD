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

#pragma once
#include "nep_file.cuh"
#include <iosfwd>
#include <string>
#include <vector>

class NEP_Model_File
{
public:
  struct ModelLayout {
    int dim_radial = 0;
    int dim_angular = 0;
    int dim = 0;
    int ann_count = 0;
    int global_parameters = 0;
    int number_of_variables_ann_1 = 0;
    int number_of_variables_ann = 0;
    int number_of_variables_descriptor = 0;
    int number_of_variables = 0;
  };

  std::string name;
  std::vector<std::string> elements;
  int num_types = 0;
  int num_neurons1 = 0;
  int num_neurons2 = 0;
  int n_max_radial = 0;
  int n_max_angular = 0;
  int basis_size_radial = 0;
  int basis_size_angular = 0;
  int L_max = 0;
  int has_q_222 = 0;
  int has_q_1111 = 0;
  int has_q_112 = 0;
  int has_q_123 = 0;
  int has_q_233 = 0;
  int has_q_134 = 0;
  bool has_multiple_cutoffs = false;
  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
  int max_radial = 0;
  int max_angular = 0;
  bool enable_zbl = false;
  bool flexible_zbl = false;
  bool use_typewise_cutoff_zbl = false;
  float zbl_rc_inner = 0.0f;
  float zbl_rc_outer = 0.0f;
  float typewise_cutoff_zbl_factor = 0.0f;
  ModelLayout layout;
  std::vector<double> energy_reference;
  std::vector<float> parameters;
  std::vector<float> q_scaler;
  std::vector<float> zbl_parameters;

  bool read(std::istream& input, std::string& error);
  bool get_md_input(std::string& input, std::string& error) const;
  static bool read_compact(
    std::istream& input,
    const std::vector<std::string>& active_species,
    NEP_File& document,
    std::string& error);

private:
  bool parse(const NEP_File& document, std::string& error);
  bool get_md_input(const std::string& tag, std::string& input, std::string& error) const;
  void compact_parameters(NEP_File& document, const std::vector<int>& active_to_full) const;
  static bool fail(std::string& error, const std::string& message, std::size_t line = 0);
  static bool read_integer(
    const NEP_File::Field& field,
    std::size_t index,
    int minimum,
    int maximum,
    int& value,
    std::string& error);
  static bool read_float(
    const NEP_File::Field& field, std::size_t index, float& value, std::string& error);
  static const NEP_File::Field* require_field(
    const NEP_File& document,
    const char* name,
    std::size_t minimum,
    std::size_t maximum,
    std::vector<std::string>& handled,
    std::string& error);
  static bool read_block(
    const NEP_File& document,
    const char* name,
    int count,
    std::vector<float>& values,
    std::vector<std::string>& handled,
    std::string& error);

  bool read_descriptor(
    const NEP_File& document, std::vector<std::string>& handled, std::string& error);
  bool read_ann(
    const NEP_File& document, std::vector<std::string>& handled, std::string& error);
  bool read_reference(
    const NEP_File& document, std::vector<std::string>& handled, std::string& error);
  bool read_zbl(
    const NEP_File& document,
    std::vector<std::string>& fields,
    std::vector<std::string>& blocks,
    std::string& error);
  void calculate_layout(int extra_inputs, int extra_outputs, int globals, int ann_count);
  bool read_parameters(
    const NEP_File& document, std::vector<std::string>& blocks, std::string& error);

  bool get_nep_md_input(std::string& input, std::string& error) const;
  void compact_nep(NEP_File& document, const std::vector<int>& active_to_full) const;
  bool read_nep(
    const NEP_File& document,
    std::vector<std::string>& fields,
    std::vector<std::string>& blocks,
    std::string& error);
  bool get_charge_md_input(std::string& input, std::string& error) const;
  void compact_charge(NEP_File& document, const std::vector<int>& active_to_full) const;
  bool read_charge(
    const NEP_File& document,
    std::vector<std::string>& fields,
    std::vector<std::string>& blocks,
    std::string& error);
  bool get_tnep_md_input(std::string& input, std::string& error) const;
  void compact_tnep(NEP_File& document, const std::vector<int>& active_to_full) const;
  bool read_tnep(
    const NEP_File& document,
    std::vector<std::string>& fields,
    std::vector<std::string>& blocks,
    std::string& error);
  void compact_vdw(NEP_File& document, const std::vector<int>& active_to_full) const;
  bool read_vdw(
    const NEP_File& document,
    std::vector<std::string>& fields,
    std::vector<std::string>& blocks,
    std::string& error);
  void compact_charge_vdw(NEP_File& document, const std::vector<int>& active_to_full) const;
  bool read_charge_vdw(
    const NEP_File& document,
    std::vector<std::string>& fields,
    std::vector<std::string>& blocks,
    std::string& error);
};
