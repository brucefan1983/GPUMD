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
