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

class Deform : public Action
{
public:
  Deform(const char** param, int num_param);

  int get_deform_x() const;
  int get_deform_y() const;
  int get_deform_z() const;
  int get_deform_xy() const;
  int get_deform_xz() const;
  int get_deform_yz() const;

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  virtual void post_integrate1(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

private:
  void parse(const char** param, int num_param);
  void parse_legacy(const char** param, int num_param);
  void parse_general(const char** param, int num_param);

  bool use_legacy_format_ = false;
  int deform_component_[6] = {0, 0, 0, 0, 0, 0};
  double deform_rate_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};
