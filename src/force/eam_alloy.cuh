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
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>

struct EAMAlloy_Data {

  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<double> F_rho_a_g;
  GPU_Vector<double> F_rho_b_g;
  GPU_Vector<double> F_rho_c_g;
  GPU_Vector<double> F_rho_d_g;
  GPU_Vector<double> rho_r_a_g;
  GPU_Vector<double> rho_r_b_g;
  GPU_Vector<double> rho_r_c_g;
  GPU_Vector<double> rho_r_d_g;
  GPU_Vector<double> phi_r_a_g;
  GPU_Vector<double> phi_r_b_g;
  GPU_Vector<double> phi_r_c_g;
  GPU_Vector<double> phi_r_d_g;
  GPU_Vector<double> d_F_rho_i_g;

  int Nelements;
  std::vector<std::string> elements_list;
  int nrho;
  double drho;
  int nr;
  double dr;
  double rc;
  std::vector<double> F_rho;
  std::vector<double> rho_r;
  std::vector<double> phi_r;
  std::vector<double> F_rho_a;
  std::vector<double> F_rho_b;
  std::vector<double> F_rho_c;
  std::vector<double> F_rho_d;
  std::vector<double> rho_r_a;
  std::vector<double> rho_r_b;
  std::vector<double> rho_r_c;
  std::vector<double> rho_r_d;
  std::vector<double> phi_r_a;
  std::vector<double> phi_r_b;
  std::vector<double> phi_r_c;
  std::vector<double> phi_r_d;
  std::vector<int> atomic_number;
  std::vector<double> atomic_mass;
  std::vector<double> lattice_constant;
  std::vector<std::string> lattice_type;
};

class EAMAlloy : public Potential
{
public:
  using Potential::compute;
  EAMAlloy(const char*, const int number_of_atoms);
  virtual ~EAMAlloy(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void initialize_eamalloy(const char*, const int);

protected:
  EAMAlloy_Data eam_data;
};
