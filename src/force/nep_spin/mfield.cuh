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

#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Spin_Magnetic_Field
{
public:
  void parse(const char** param, int num_param, const std::vector<Group>& group);
  bool enabled() const { return mfield_enabled_; }
  void compute(
    const Box& box, const GPU_Vector<double>& position_per_atom,
    const std::vector<Group>& group, const GPU_Vector<double>& spin_per_atom,
    GPU_Vector<double>& potential_per_atom, GPU_Vector<double>& mforce_per_atom) const;

private:
  int mfield_group_method_ = 0;
  int mfield_group_id_ = 0;
  bool mfield_use_region_ = false;
  double mfield_region_[6] = {};
  bool mfield_enabled_ = false;
  double mfield_[3] = {}; // eV/muB
};
