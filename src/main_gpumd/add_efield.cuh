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

#include <vector>

class Atom;
class Group;

class Add_Efeild
{
public:

  void parse(const char** param, int num_param, const std::vector<Group>& group);
  void compute(const int step, const std::vector<Group>& groups, Atom& atom);
  void finalize();

private:

  int num_calls_ = 0;
  int table_length_[10];
  std::vector<double> efield_table_[10];
  int grouping_method_[10];
  int group_id_[10];
};
