/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Atom;

class Dump_Beads
{
public:
  void parse(const char** param, int num_param);
  void preprocess(const int number_of_atoms, const int number_of_beads);
  void process(const int step, const double global_time, const Box& box, Atom& atom);
  void postprocess();

private:
  bool dump_ = false;
  int dump_interval_ = 1;
  int has_velocity_ = 0;
  int has_force_ = 0;
  int number_of_beads_ = 0;
  std::vector<FILE*> fid_;
  void output_line2(FILE* fid, const double time, const Box& box);
  std::vector<double> cpu_position_;
  std::vector<double> cpu_velocity_;
  std::vector<double> cpu_force_;
};
