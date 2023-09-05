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

#include "mc_ensemble.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include <memory>
#include <vector>

class Atom;

class MC
{
public:
  std::unique_ptr<MC_Ensemble> mc_ensemble;

  void initialize(void);
  void finalize(void);
  void compute(int step, int num_steps, Atom& atom, Box& box, std::vector<Group>& group);

  void parse_mc(
    const char** param, int num_param, std::vector<Group>& group, std::vector<int>& cpu_type);

private:
  bool do_mcmd = false;
  int num_steps_md = 0;
  int num_steps_mc = 0;
  int grouping_method = 0;
  int group_id = 0;
  double temperature_initial = 0.0;
  double temperature_final = 0.0;
};
