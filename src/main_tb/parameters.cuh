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

#include <limits>
#include <string>
#include <vector>

class Parameters
{
public:
  explicit Parameters(const std::string& input_file = "tb.in");

  std::vector<std::string> elements;
  int num_types = 0;
  std::string basis = "sp3";
  double rc_radial = 5.0;
  double rc_angular = 4.0;
  double rc_hopping = 5.0;

  int n_max_radial = 6;
  int n_max_angular = 6;
  int basis_size_radial = 6;
  int basis_size_angular = 6;
  int L_max = 4;
  int num_neurons_onsite = 30;
  int num_neurons_hopping = 30;

  int orbitals_per_atom;
  int num_onsite_outputs;
  int num_hopping_outputs_homo;
  int num_hopping_outputs_hetero;
  int max_hopping_outputs;

  int dim_radial;
  int dim_angular;
  int dim;
  int num_c_radial;
  int num_c_angular;
  int descriptor_block_size;

  int offset_onsite;
  int offset_hopping;
  int offset_descriptor;
  int onsite_block_size;
  int number_of_variables_onsite;
  int number_of_variables_hopping;
  int number_of_variables_descriptor;
  int number_of_variables;

  int num_element_pairs;
  std::vector<int> hopping_pair_offsets;
  std::vector<int> hopping_pair_outputs;

  int prediction = 0;
  int maximum_epoch = 100;
  int seed = 2024;

  double learning_rate = 1.0e-3;
  double weight_decay = 0.0;

  bool lr_scheduler_explicit = false;
  double lr_factor = 0.5;
  int lr_patience = 5;
  double lr_rel_threshold = 3.0e-4;
  double lr_min = 1.0e-6;

  int structure_batch = 1;
  int k_batch = 32;

  int band_min = 0;
  int band_max = 6;
  int band_curriculum = 1;
  std::vector<double> band_weights;
  double band_weight_sum;

  double dispersion_weight = 0.0;
  double dispersion_sigma = std::numeric_limits<double>::infinity();

  bool is_sp3() const { return basis == "sp3"; }
  bool is_sp3sstar() const { return basis == "sp3sstar"; }
  bool is_sp3d5() const { return basis == "sp3d5"; }
  bool is_sp3d5sstar() const { return basis == "sp3d5sstar"; }
  bool has_d_shell() const { return is_sp3d5() || is_sp3d5sstar(); }
  bool has_sstar_shell() const { return is_sp3sstar() || is_sp3d5sstar(); }

  int hopping_outputs_for_pair(int type_a, int type_b) const;

  void print() const;

private:
  struct BandWeightRange {
    int band_min = 0;
    int band_max = 0;
    double weight = 1.0;
  };
  std::vector<BandWeightRange> band_weight_ranges_;

  int unordered_type_pair_index(int type_a, int type_b) const;
  int hopping_block_size(int num_outputs) const;
  void read_input(const std::string& filename);
  void parse_keyword(const std::vector<std::string>& tokens, const std::string& location);
  void validate() const;
  void derive_model_sizes();
  void derive_band_weights();
};
