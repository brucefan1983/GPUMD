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
#include <stdio.h>

// ADP (Angular Dependent Potential) implementation
// Reference: Y. Mishin et al., Acta Mater. 53, 4029 (2005)

struct ADP_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> NL_shift;       // encoded periodic image shifts per neighbor entry
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<double> Fp;          // derivative of embedding function F'(rho)
  GPU_Vector<double> mu;          // dipole distortion terms (3 components per atom)
  GPU_Vector<double> lambda;      // quadruple distortion terms (6 components per atom: xx,yy,zz,yz,xz,xy)
  
  GPU_Vector<int> pair_index_map_g;      // Lookup table for pair indices (size Nelements*Nelements)
  
  // Tabulated functions from ADP file
  int Nelements;
  std::vector<std::string> elements_list;
  int nrho;
  double drho;
  int nr;
  double dr;
  double rc;
  double inv_drho = 0.0;
  double inv_dr = 0.0;
  
  // EAM-like functions
  std::vector<double> F_rho;      // embedding function F(rho)
  std::vector<double> rho_r;      // electron density function rho(r)
  std::vector<double> phi_r;      // pair potential r*phi(r)
  
  // ADP-specific functions
  std::vector<double> u_r;        // u(r) function for dipole term
  std::vector<double> w_r;        // w(r) function for quadruple term
  
  // Spline coefficients for interpolation
  std::vector<double> F_rho_a, F_rho_b, F_rho_c, F_rho_d;
  std::vector<double> rho_r_a, rho_r_b, rho_r_c, rho_r_d;
  std::vector<double> phi_r_a, phi_r_b, phi_r_c, phi_r_d;
  std::vector<double> u_r_a, u_r_b, u_r_c, u_r_d;
  std::vector<double> w_r_a, w_r_b, w_r_c, w_r_d;
  
  // GPU versions of spline coefficients
  GPU_Vector<double> F_rho_a_g, F_rho_b_g, F_rho_c_g, F_rho_d_g;
  GPU_Vector<double> rho_r_a_g, rho_r_b_g, rho_r_c_g, rho_r_d_g;
  GPU_Vector<double> phi_r_a_g, phi_r_b_g, phi_r_c_g, phi_r_d_g;
  GPU_Vector<double> u_r_a_g, u_r_b_g, u_r_c_g, u_r_d_g;
  GPU_Vector<double> w_r_a_g, w_r_b_g, w_r_c_g, w_r_d_g;
};

class ADP : public Potential
{
public:
  using Potential::compute;
  ADP(const char* file_potential, const int number_of_atoms);
  virtual ~ADP(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void initialize(const char* file_potential, const int number_of_atoms);
  void ensure_capacity(int number_of_atoms);

protected:
  ADP_Data adp_data;
  
  // File reading and initialization
  void read_adp_file(const char* file_potential);
  void setup_spline();
  void calculate_spline(
    const double* y, double dx,
    double* a, double* b, double* c, double* d,
    int n_functions, int n_points);
};
