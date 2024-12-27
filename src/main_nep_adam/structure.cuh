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
#include <vector>

class Parameters;

struct Structure {
  int num_cell[3];
  int num_atom;
  int has_virial;
  int has_temperature;
  double weight;
  double energy;
  double virial[6];
  double box_original[9];
  double volume;
  double box[18];
  double temperature;
  std::vector<int> type;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
};

bool read_structures(bool is_train, Parameters& para, std::vector<Structure>& structures);
