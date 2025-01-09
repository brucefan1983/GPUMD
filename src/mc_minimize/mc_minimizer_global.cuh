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

class MC_Minimizer_Global : public MC_Minimizer
{
private:
  double temperature;
  double force_tolerance;
  double max_relax_steps;
  double energy_last_step = 0;
public:
    MC_Minimizer_Global(const char** param, int num_param,
    double temperature_input,
    double force_tolerance_input,
    int max_relax_steps_input);
    ~MC_Minimizer_Global();

    virtual void compute(
    int trials,
    Force& force,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);
};

