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

class Parameters;

struct Structure {
  int index_in_file = 0; // position of the structure in train.xyz or test.xyz
  int pbc = 1; // 1 for PPP and 0 for FFF in long-range training
  int num_cell[3];
  int num_atom;
  int has_virial;
  int has_atomic_virial;
  int atomic_virial_diag_only;
  int has_bec;
  int has_temperature;
  float weight;
  float charge = 0.0f;
  float energy = 0.0f;
  double energy_total = 0.0; // the reference total energy, for the combinations of ediff.in
  float energy_weight = 1.0f;
  float virial[6];
  float box_original[9];
  float volume;
  float box[18];
  float temperature;
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

bool read_structures(bool is_train, Parameters& para, std::vector<Structure>& structures);
