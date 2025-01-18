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

#include "mc_minimizer.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include <memory>
#include <vector>

class MC_Minimize
{
private:

    void parse_group(const char** param, int num_param, std::vector<Group>& groups, int num_param_before_group);
    int grouping_method = -1;
    int group_id = -1;

    //parameters
    int num_trials_mc = 0; 
    double temperature = 0; 
    double force_tolerance = 0;
    int max_relax_steps = 0;
    double scale_factor = 0;


public:
    void parse_mc_minimize(const char** param, int num_param, std::vector<Group>& group, Atom& atom, Box& box, Force& force);

    void compute(
      Force& force,
      Atom& atom,
      Box& box,
      std::vector<Group>& group);

    std::unique_ptr<MC_Minimizer> mc_minimizer;
};

