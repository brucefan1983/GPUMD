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
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>

class NEP_Parameters;

void read_nep_model(
  const std::string& filename, NEP_Parameters& para, float* parameters);

bool read_nep_restart(
  const char* filename,
  const NEP_Parameters& para,
  std::vector<float>& mu,
  std::vector<float>& sigma);

void read_nep_foundation_restart(
  const std::string& filename, std::vector<float>& mu, std::vector<float>& sigma);

void write_nep_restart(
  const char* filename,
  const NEP_Parameters& para,
  const std::vector<float>& mu,
  const std::vector<float>& sigma);

void read_nep_q_scaler(NEP_Parameters& para);

void write_nep_model(
  FILE* file, NEP_Parameters& para, const float* parameters, int max_radial, int max_angular);

void check_existing_nep_model(NEP_Parameters& para);

// Shared structural codec; model-specific metadata precedes this section.
struct NepTxtHeader {
  std::string model_token;
  int version;
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

bool read_nep_structural_header(
  std::ifstream& input, const std::string& filename, NepTxtHeader& header, std::string& error);
void write_nep_model_body(
  FILE* file, const NEP_Parameters& para, const float* parameters, int max_radial, int max_angular);
