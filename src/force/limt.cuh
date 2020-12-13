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

struct LIMT_Data {
  GPU_Vector<double> b;    // bond orders
  GPU_Vector<double> bp;   // derivative of bond orders
  GPU_Vector<double> f12x; // partial forces
  GPU_Vector<double> f12y;
  GPU_Vector<double> f12z;
  GPU_Vector<int> NN_short; // for many-body part
  GPU_Vector<int> NL_short; // for many-body part
};

struct LIMT_Para {
  double a[3];
  double b[3];
  double lambda[3];
  double mu[3];
  double n[3];
  double beta[3];
  double h[3];
  double r1[3];
  double r2[3];
  double gamma[3];
  double pi_factor[3];
  double minus_half_over_n[3];
};

class LIMT : public Potential
{
public:
  LIMT(FILE*, int, const Neighbor& neighbor);
  virtual ~LIMT(void);
  virtual void compute(
    const int type_shift,
    const Box& box,
    const Neighbor& neighbor,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

protected:
  int num_types; // number of atom tpyes
  LIMT_Data LIMT_data;
  LIMT_Para para;
};
