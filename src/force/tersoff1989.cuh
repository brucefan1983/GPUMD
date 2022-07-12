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
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <stdio.h>

struct Tersoff1989_Parameters {
  double a, b, lambda, mu, beta, n, c, d, c2, d2, h, r1, r2;
  double pi_factor, one_plus_c2overd2, minus_half_over_n;
};

struct Tersoff1989_Data {
  GPU_Vector<double> b;    // bond orders
  GPU_Vector<double> bp;   // derivative of bond orders
  GPU_Vector<double> f12x; // partial forces
  GPU_Vector<double> f12y;
  GPU_Vector<double> f12z;
  GPU_Vector<int> NN, NL; // neighbor list for angular-dependent potentials
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

class Tersoff1989 : public Potential
{
public:
  Tersoff1989(FILE*, int sum_of_types, const int num_atoms);
  virtual ~Tersoff1989(void);
  virtual void compute(
    const int type_shift,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

protected:
  Tersoff1989_Parameters ters0;
  Tersoff1989_Parameters ters1;
  Tersoff1989_Parameters ters2;
  Tersoff1989_Data tersoff_data;
};
