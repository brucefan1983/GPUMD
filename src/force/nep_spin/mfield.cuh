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

#include "measure/action.cuh"
#include <vector>

class Spin_Magnetic_Field : public Action
{
public:
  Spin_Magnetic_Field(const std::vector<std::string>& tokens, const std::vector<Group>& group);

  void pre_run(
    int number_of_steps,
    double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;
  void setup_force(
    double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;
  void post_force(
    int step,
    double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

private:
  void parse(const char** param, int num_param, const std::vector<Group>& group);
  void compute(Box& box, const std::vector<Group>& group, Atom& atom) const;
  GPU_Vector<int> active_types_;
  int mfield_group_method_ = 0;
  int mfield_group_id_ = 0;
  bool mfield_use_region_ = false;
  double mfield_region_[6] = {};
  bool mfield_enabled_ = false;
  double mfield_[3] = {}; // eV/muB
};
