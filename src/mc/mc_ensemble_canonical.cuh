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
#include "mc_ensemble.cuh"

class MC_Ensemble_Canonical : public MC_Ensemble
{
public:
  MC_Ensemble_Canonical(const char** param, int num_param, int num_steps_mc);
  virtual ~MC_Ensemble_Canonical(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

  virtual void compute_local(
    double scale_factor,
    double temperature,
    Force& force,
    int max_relaxation_step,
    double force_tolerance,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

private:
  GPU_Vector<int> NN_ij;
  GPU_Vector<int> NL_ij;
};
