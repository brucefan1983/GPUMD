/*
    Copyright 2019 Zheyong Fan
    This file is part of GPUGA.
    GPUGA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUGA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUGA.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "gpu_vector.cuh"
#include <vector>

class Neighbor;

class Potential
{
public:
  virtual ~Potential() = default;
  virtual void initialize(int, int) = 0;
  virtual void update_potential(const std::vector<float>&) = 0;
  virtual void find_force(
    int,
    int,
    int*,
    int*,
    int,
    int*,
    float*,
    Neighbor*,
    float*,
    GPU_Vector<float>&,
    GPU_Vector<float>&,
    GPU_Vector<float>&) = 0;
};
