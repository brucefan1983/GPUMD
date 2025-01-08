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

  void parse_mc(const char** param, int num_param, std::vector<Group>& group, Atom& atom);

  void parse_mc_local(const char** param, int num_param, std::vector<Group>& group, Atom& atom, Box& box, Force& force);


private:
  bool do_mcmd = false;
  int num_steps_md = 0;
  int num_steps_mc = 0;
  int num_types_mc = 0;
  int grouping_method = -1;
  int group_id = -1;
  double temperature_initial = 0.0;
  double temperature_final = 0.0;
  double kappa = 0.0;
  std::vector<std::string> species;
  std::vector<int> types;
  std::vector<int> num_atoms_species;
  std::vector<double> mu_or_phi;

  void parse_group(
    const char** param, int num_param, std::vector<Group>& groups, int num_param_before_group);
  void check_species_canonical(std::vector<Group>& groups, Atom& atom);
  void check_species_sgc(std::vector<Group>& groups, Atom& atom);
};
