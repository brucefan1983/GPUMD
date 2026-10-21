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

class NEP_Parameters;
class Dataset;
struct XYZ_Record;

struct Structure {
  int index_in_file = 0; // position of the structure in train.xyz or test.xyz
  int pbc = 1; // 1 for PPP and 0 for FFF in long-range training
  int num_cell[3];
  int num_atom;
  int has_virial = 0;
  float weight;
  double energy = 0.0;
  double energy_total = 0.0; // the reference total energy, for the combinations of ediff.in
  float energy_weight = 1.0f;
  float virial[6];
  float box_original[9];
  float volume;
  float box[18];
  std::vector<int> type;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  std::string name; // lowercase label from the optional name= field
};

// The default reader handles energy, force and virial labels. Models extend it
// locally for their own inputs and batch data.
// The reader is needed only while constructing the training and test data.
class NEP_Data_Reader
{
public:
  explicit NEP_Data_Reader(NEP_Parameters& para) : para(para) {}
  virtual ~NEP_Data_Reader() = default;
  bool read_structures(bool is_train, std::vector<Structure>& structures);
  // Preparation hooks run before the reader switches to the next XYZ file.
  // Model labels use index_in_file to survive sorting and ediff grouping.
  virtual bool uses_grouped_batches() const { return false; }
  virtual void prepare_train(std::vector<Structure>& structures, int batch_size,
    int& num_batches, std::vector<int>& batch_sizes) {}
  virtual void prepare_test(const std::vector<Structure>& structures, bool has_test_set) {}
  virtual void prepare_dataset(Dataset& dataset) {}

protected:
  NEP_Parameters& para;
  virtual void begin_reading() {}
  virtual void read_structure(const XYZ_Record& record, Structure& structure);
  virtual void print_summary(const std::vector<Structure>& structures) const;
  void read_geometry(const XYZ_Record& record, Structure& structure) const;
  void read_energy(const XYZ_Record& record, Structure& structure, bool required) const;
};

// Boundary handling shared by the long-range models.
void read_nep_boundaries(const XYZ_Record& record, Structure& structure);
void print_nep_boundaries(const std::vector<Structure>& structures);
