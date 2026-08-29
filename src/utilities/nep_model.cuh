/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version. GPUMD is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have
   received a copy of the GNU General Public License along with GPUMD.  If not, see
   <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <string>
#include <vector>

bool is_compactable_nep_model(const std::string& file_potential);
std::vector<std::string> get_potential_atom_symbols(const std::string& file_potential);
void initialize_nep_active_species(
  const std::string& file_potential, const std::vector<std::string>& atom_symbols);
const std::vector<std::string>& get_nep_active_species();
std::vector<int> get_nep_active_to_full(const std::vector<std::string>& atom_symbols_full);
int get_nep_active_type(const std::string& atom_symbol);

std::vector<float> compact_nep_parameters(
  const std::vector<float>& parameters_full,
  int num_types_full,
  const std::vector<int>& active_to_full,
  int ann_type_size,
  int ann_global_size,
  int ann_blocks,
  int n_max_radial,
  int basis_size_radial,
  int n_max_angular,
  int basis_size_angular,
  int tail_size);

std::vector<float> compact_nep_zbl_parameters(
  const std::vector<float>& parameters_full,
  int num_types_full,
  const std::vector<int>& active_to_full);
