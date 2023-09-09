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

class MC_Ensemble_SGC : public MC_Ensemble
{
public:
  MC_Ensemble_SGC(
    int num_steps_mc,
    bool is_vcsgc,
    std::vector<std::string>& species,
    std::vector<int>& types,
    std::vector<double>& mu_or_phi,
    double kappa);
  virtual ~MC_Ensemble_SGC(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

private:
  GPU_Vector<int> NN_ij;
  GPU_Vector<int> NL_ij;
  bool is_vcsgc = false;
  std::vector<std::string> species;
  std::vector<int> types;
  std::vector<double> mu_or_phi;
  double kappa;
};
