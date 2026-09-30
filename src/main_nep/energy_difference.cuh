/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <string>
#include <vector>

class Dataset;
class Parameters;
struct Structure;

// One term of a line of ediff.in, with the name in lowercase.
struct EnergyDiffTerm {
  std::string name;
  double coefficient;
};

// One line of ediff.in, a linear combination of the total energies of named structures.
struct EnergyDiffEntry {
  std::vector<EnergyDiffTerm> terms;
  int line_number;
  float weight = 1.0f;
};

// A linear combination of the total energies of structures of one data set that enters the loss.
// All structures of a combination lie in one batch.
struct EnergyDiffCombination {
  int batch;                       // batch that holds all structures
  std::vector<int> local;          // index of each structure within the batch
  std::vector<double> coefficient; // coefficient of each structure
  double ref_total_eV;             // the same combination of the reference total energies
  float weight;
};

// The loss term over the linear combinations of the total energies of named structures listed in
// ediff.in, which the keyword lambda_d activates.
class EnergyDifference
{
public:
  // Reads the optional name= field of a comment line of train.xyz or test.xyz, split into tokens
  // in lowercase, and returns the name, or an empty string without the field.
  static std::string read_structure_name(
    const std::vector<std::string>& tokens, const std::string& xyz_filename, const int line_number);
  // Reads ediff.in in training mode when lambda_d is set, keeps the structures of each training
  // combination in one batch, and resolves the training combinations. batch_size is the batch size
  // of nep.in, and num_batches and batch_sizes describe the batches, which the grouping revises.
  void read_train(
    Parameters& para,
    std::vector<Structure>& structures_train,
    const int batch_size,
    int& num_batches,
    std::vector<int>& batch_sizes);
  // Resolves the test combinations and prints the summary of ediff.in.
  void read_test(
    const Parameters& para, const std::vector<Structure>& structures_test, const bool has_test_set);
  // Weighted RMSE of the training combinations in batch batch_id, which the caller has evaluated.
  float get_rmse_train(Dataset& dataset, const int batch_id, const int device_id) const;
  // Weighted RMSE of the test combinations, which the caller has evaluated.
  float get_rmse_test(Dataset& dataset, const int device_id) const;

private:
  std::vector<EnergyDiffEntry> entries;
  std::vector<bool> is_entry_in_train;
  std::vector<bool> is_entry_in_test;
  std::vector<EnergyDiffCombination> combinations_train;
  std::vector<EnergyDiffCombination> combinations_test;
};
