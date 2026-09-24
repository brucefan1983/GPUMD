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
#include "ensemble_lan.cuh"
#include "langevin_utilities.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <map>
#include <math.h>
#include <string>
#include <vector>

class Ensemble_TI : public Ensemble_LAN
{
public:
  Ensemble_TI(const std::vector<std::string>& tokens);
  ~Ensemble_TI(void) override;

  void finalize_run(const Atom& atom, const Box& box) override;

  void initialize_before_first_step(
    const double time_step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  void compute2(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atoms,
    GPU_Vector<double>& thermo,
    Force& force) override;

  void find_thermo(
    const Box& box,
    const std::vector<Group>& group,
    const Atom& atom,
    GPU_Vector<double>& thermo);
  double get_espring_sum(const int number_of_atoms);
  void add_spring_force(const Box& box, Atom& atom);
  void init(const Atom& atom, const GPU_Vector<double>& thermo);

protected:
  FILE* output_file = nullptr;
  double lambda = 0;
  double pe, espring;
  // spring constants
  std::map<std::string, double> spring_map;
  GPU_Vector<double> gpu_k;
  std::vector<double> cpu_k;
  GPU_Vector<double> gpu_espring;
  GPU_Vector<double> position_0;
  std::vector<double> thermo_cpu;

private:
  void close_output_file(const bool print_message);
};
