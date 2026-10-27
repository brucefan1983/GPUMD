/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include <iosfwd>
#include <string>
#include <vector>

// Current Spin model data. Parameter order is identical to the training arrays
// and the @ parameters block; no legacy text or layout conversion is involved.
struct NEP_Spin_Model_File {
  static constexpr int MAX_DIM_SPIN = 199; // structural 103 + magnetic 96
  std::vector<std::string> elements;
  float rc_radial = 0, rc_angular = 0;
  int max_radial = 0, max_angular = 0;
  int n_max[2] = {}, basis_size[2] = {}, l_max[7] = {};
  int num_neurons = 0;
  int spin_basis_size = 0, spin_l_max = 0, spin_compress = 0;
  int spin_order = 0, spin_soc = 0, spin_projection_size = 0;
  int dim_struct = 0, dim_spin = 0, dim = 0;
  std::vector<float> spin_cutoff;
  std::vector<int> spin_type;
  bool enable_zbl = false, use_typewise_cutoff_zbl = false;
  float zbl_rc_inner = 0, zbl_rc_outer = 0, typewise_cutoff_zbl_factor = 0;
  std::vector<double> energy_reference;
  std::vector<float> parameters, q_scaler;

  bool read(std::istream& input, std::string& error);
  static int spin_dimension(int channels, int l_max, int order, int soc);
};
