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
#include "ensemble.cuh"
#include <random>

class Ensemble_NPT_SCR : public Ensemble
{
public:
  Ensemble_NPT_SCR(const char** param, int num_param, const Box& box);
  virtual ~Ensemble_NPT_SCR(void);

  double get_temperature1() const;
  double get_temperature2() const;
  int get_num_target_pressure_components() const;

  virtual void initialize_run(
    const double time_step, Atom& atom, Box& box, const std::vector<Group>& group);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

protected:
  std::mt19937 rng;
  void initialize_rng();

private:
  double temperature1_ = 0.0;
  double temperature2_ = 0.0;
};
