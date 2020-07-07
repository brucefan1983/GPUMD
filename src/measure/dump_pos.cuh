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

/*----------------------------------------------------------------------------80
Parent class for position dumping
------------------------------------------------------------------------------*/

#pragma once
#ifndef DUMP_POS_H
#define DUMP_POS_H

#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class DUMP_POS
{
public:
  int interval; // output interval
  char file_position[200];
  int precision; // 0 = normal output, 1 = single precision, 2 = double
  virtual void initialize(char* input_dir, const int number_of_atoms) = 0;
  virtual void finalize() = 0;
  virtual void dump(
    const int step,
    const double global_time,
    const Box& box,
    const std::vector<int>& cpu_type,
    GPU_Vector<double>& position_per_atom,
    std::vector<double>& cpu_position_per_atom) = 0;

  DUMP_POS() {}
  virtual ~DUMP_POS() {}
};

#endif // DUMP_POS
