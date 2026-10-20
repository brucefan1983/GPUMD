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
struct XYZ_Record;

struct Structure {
  int index_in_file = 0; // position of the structure in train.xyz or test.xyz
  int pbc = 1; // 1 for PPP and 0 for FFF in long-range training
  int num_cell[3];
  int num_atom;
  int has_virial = 0;
  int has_atomic_virial = 0;
  int atomic_virial_diag_only = 0;
  int has_bec = 0;
  int has_temperature = 0;
  float weight;
  float charge = 0.0f;
  float energy = 0.0f;
  double energy_total = 0.0; // the reference total energy, for the combinations of ediff.in
  float energy_weight = 1.0f;
  float virial[6];
  float box_original[9];
  float volume;
  float box[18];
  float temperature = 0.0f;
  std::vector<int> type;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  std::vector<float> avirialxx;
  std::vector<float> avirialyy;
  std::vector<float> avirialzz;
  std::vector<float> avirialxy;
  std::vector<float> avirialyz;
  std::vector<float> avirialzx;
  std::vector<float> bec;
  std::string name; // lowercase label from the optional name= field
};

// The default reader handles energy, force and virial labels. Models extend it
// locally for their own inputs and for the states distinguished by ediff.in.
// The reader is needed only while constructing the training and test data.
class NEP_Data_Reader
{
public:
  explicit NEP_Data_Reader(Parameters& para) : para(para) {}
  virtual ~NEP_Data_Reader() = default;
  bool read_structures(bool is_train, std::vector<Structure>& structures) const;
  // Return the reason to reject a pair in ediff.in, or nullptr if it is allowed.
  virtual const char* energy_difference_conflict(
    const Structure& a, const Structure& b) const;

protected:
  Parameters& para;
  virtual void read_structure(const XYZ_Record& record, Structure& structure) const;
  virtual void print_summary(const std::vector<Structure>& structures) const;
  void read_geometry(const XYZ_Record& record, Structure& structure) const;
  void read_energy(const XYZ_Record& record, Structure& structure, bool required) const;
};

// Boundary handling shared by the long-range models.
void read_nep_boundaries(const XYZ_Record& record, Structure& structure);
void print_nep_boundaries(const std::vector<Structure>& structures);
