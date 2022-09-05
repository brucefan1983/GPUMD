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

struct Tersoff_mini_Data {
  GPU_Vector<float> b;    // bond orders
  GPU_Vector<float> bp;   // derivative of bond orders
  GPU_Vector<float> f12x; // partial forces
  GPU_Vector<float> f12y;
  GPU_Vector<float> f12z;
  GPU_Vector<int> NN, NL; // neighbor list for angular-dependent potentials
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

struct Tersoff_mini_Para {
  float a[3];
  float b[3];
  float lambda[3];
  float mu[3];
  float beta[3];
  float n[3];
  float h[3];
  float r1[3];
  float r2[3];
  float pi_factor[3];
  float minus_half_over_n[3];
};

class Tersoff_mini : public Potential
{
public:
  Tersoff_mini(FILE*, int, const int num_atoms);
  virtual ~Tersoff_mini(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

protected:
  int num_types; // number of atom tpyes
  Tersoff_mini_Data tersoff_mini_data;
  Tersoff_mini_Para para;
};
