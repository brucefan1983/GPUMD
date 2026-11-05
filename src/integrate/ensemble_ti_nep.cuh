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
#include "force/force.cuh"
#include "langevin_utilities.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <math.h>
#include <string>
#include <vector>

class Ensemble_TI_Nep : public Ensemble_LAN
{
public:
  Ensemble_TI_Nep(const std::vector<std::string>& tokens);
  ~Ensemble_TI_Nep(void) override;

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
  double get_pe_nep2(const int number_of_atoms);
  void add_nep2_force(Atom& atom);
  void init(const int number_of_steps, const Atom& atom, const GPU_Vector<double>& thermo);
  void find_lambda(
    const int step,
    const Box& box,
    const std::vector<Group>& group,
    Atom& atom,
    GPU_Vector<double>& thermo);
  double switch_func(double t);
  double dswitch_func(double t);

protected:
  FILE* output_file = nullptr;
  double lambda = 0, dlambda = 0;
  int t_equil = -1, t_switch = -1;
  double pe, pe_nep2;
  double F_diff = 0; // Helmholtz free energy difference F(NEP1) - F(NEP2) (eV/atom)
  bool auto_switch = true;
  bool potentials_checked = false;
  // per-atom properties computed with the second potential
  GPU_Vector<double> potential_nep2;
  GPU_Vector<double> force_nep2;
  GPU_Vector<double> virial_nep2;
  std::vector<double> thermo_cpu;

private:
  void close_output_file(const bool print_message);
};
