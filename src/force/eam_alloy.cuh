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
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>

struct EAMAlloy_Data {

  GPU_Vector<float4> F_rho_g;    // size = Nelements * nrho
  GPU_Vector<float4> rho_r_g;    // size = Nelements * nr
  GPU_Vector<float4> phi_r_g;    // size = Nelements * Nelements * nr   (r*phi)
  GPU_Vector<float> d_F_rho_i_g; // dF/drho per atom

  int Nelements;
  std::vector<std::string> elements_list;
  int nrho;
  float drho;
  int nr;
  float dr;
  float rc;
};

class EAMAlloy : public Potential
{
public:
  using Potential::compute;

  EAMAlloy(const char*, const int number_of_atoms, const int max_neighbor = 400);
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
  Neighbor neighbor;
};
