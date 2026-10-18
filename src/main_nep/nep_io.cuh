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
#include <string>
#include <vector>

class Parameters;

// the model hyperparameters as recorded in the header of a nep.txt file
struct NepTxtHeader {
  std::string model_token;
  int version;
  int model_type;
  int charge_mode;
  int vdw;
  int charge_vdw;
  bool enable_zbl;
  bool flexible_zbl;
  bool use_typewise_cutoff_zbl;
  float zbl_rc_inner;
  float zbl_rc_outer;
  float typewise_cutoff_zbl_factor;
  int num_types;
  std::vector<std::string> elements;
  bool has_multiple_cutoffs;
  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
  int n_max_radial;
  int n_max_angular;
  int basis_size_radial;
  int basis_size_angular;
  int L_max;
  int has_q_222; // as written to nep.txt, that is 2 or 0
  int has_q_1111;
  int has_q_112;
  int has_q_123;
  int has_q_233;
  int has_q_134;
  int num_neurons1;
  int num_neurons2;
  int number_of_header_lines; // 6 without a zbl line, 7 with one
};

// parse the header of a nep.txt file; returns false and sets error if it cannot be parsed
bool read_nep_txt_header(const std::string& filename, NepTxtHeader& header, std::string& error);

void check_existing_nep_model(Parameters& para);
