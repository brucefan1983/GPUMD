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
#include <string>
#include <vector>

class NEP3
{
public:
  struct ParaMB {
    int version = 2;
    double rc_radial = 0.0;
    double rc_angular = 0.0;
    double rcinv_radial = 0.0;
    double rcinv_angular = 0.0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int L_max = 0;
    int dim_angular;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    double q_scaler[140];
  };

  struct ANN {
    int dim = 0;
    int num_neurons1 = 0;
    int num_para = 0;
    const double* w0;
    const double* b0;
    const double* w1;
    const double* b1;
    const double* c;
  };

  struct ZBL {
    bool enabled = false;
    double rc_inner = 1.0;
    double rc_outer = 2.0;
    double atomic_numbers[10];
  };

  NEP3(const int N, const std::string& potential_filename);
  void compute(
    const std::vector<int>& NN_radial,
    const std::vector<int>& NL_radial,
    const std::vector<int>& NN_angular,
    const std::vector<int>& NL_angular,
    const std::vector<int>& type,
    const std::vector<double>& r12,
    std::vector<double>& potential_per_atom,
    std::vector<double>& force_per_atom,
    std::vector<double>& virial_per_atom);
  void find_descriptor(
    const std::vector<int>& NN_radial,
    const std::vector<int>& NL_radial,
    const std::vector<int>& NN_angular,
    const std::vector<int>& NL_angular,
    const std::vector<int>& type,
    const std::vector<double>& r12,
    std::vector<double>& descriptor);

  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  std::vector<double> Fp;
  std::vector<double> sum_fxyz;
  std::vector<double> parameters;
  void update_potential(const double* parameters, ANN& ann);
};
