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

#include "property.cuh"
#include <vector>

class Add_Efield : public Property
{
public:
  Add_Efield(const char** param, int num_param, const std::vector<Group>& group);

  void setup_force(
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

private:
  void apply_field(
    const int md_step,
    std::vector<Group>& group,
    Atom& atom,
    Force& force);

  int table_length_ = 0;
  std::vector<double> efield_table_;
  int grouping_method_ = 0;
  int group_id_ = 0;
  bool is_nep_charge_ = false;
  bool use_bec_ = false;
};
