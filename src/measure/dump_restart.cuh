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

#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>
class Box;
class Group;

class Dump_Restart
{
public:
  void parse(const char** param, int num_param);
  void preprocess();
  void process(
    const int step,
    const Box& box,
    const std::vector<Group>& group,
    const std::vector<std::string>& cpu_atom_symbol,
    const std::vector<int>& cpu_type,
    const std::vector<double>& cpu_mass,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    std::vector<double>& cpu_position_per_atom,
    std::vector<double>& cpu_velocity_per_atom);
  void postprocess();

private:
  bool dump_ = false;
  int dump_interval_ = 1;
};
