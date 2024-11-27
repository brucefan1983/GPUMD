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

#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string>
#include <vector>

class Extrapolation
{
public:
  void parse(const char** params, int num_params);
  void allocate_memory(Force& force, Atom& atom, Box& box);
  void calculate_gamma();
  void process(int step);
  void dump();
  void output_line2();
  FILE* f;
  std::vector<GPU_Vector<double>*> asi_data;
  std::map<int, double*> asi;
  int B_size_per_atom;
  // points to potential
  GPU_Vector<double> B;
  // max gamma
  GPU_Vector<double> gamma;
  std::vector<double> gamma_cpu;
  double max_gamma;
  Atom* atom;
  Box* box;
  bool activated = false;
  int check_interval = 1;
  int dump_interval = 1;
  int last_dump = 0;
  double gamma_low = 0;
  double gamma_high = 1e100;

private:
  void load_asi(std::string asi_file_name);
};
