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
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
class Dataset;
struct TrainingReport;
struct NEP_Energy_Error;

void print_nep_energy_header(bool has_ediff);
void print_nep_charge_header(bool has_ediff);
void write_nep_energy_error(FILE* file, const TrainingReport& report,
  const NEP_Energy_Error& train, const NEP_Energy_Error& test, bool has_ediff);
void write_nep_charge_error(FILE* file, const TrainingReport& report,
  const NEP_Energy_Error& train, const NEP_Energy_Error& test,
  float charge_train, float bec_train, float charge_test, float bec_test, bool has_ediff);

void copy_nep_energy_predictions(Dataset& data);
void write_nep_structure_predictions(bool is_stress, int num_components, FILE* fid,
  float* prediction, float* reference, Dataset& dataset, int nc);
void write_nep_atomic_predictions(int num_components, FILE* fid, float* prediction,
  float* reference, Dataset& dataset, int nc);
void write_nep_energy_predictions(
  const std::string& label, const std::vector<std::pair<Dataset*, int>>& structures);
