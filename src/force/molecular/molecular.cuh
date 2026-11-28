/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "force/potential.cuh"
#include "utilities/gpu_scan.cuh"

class Atom;

class Molecular : public Potential
{
public:
  Molecular(const char* forcefield, const char* topology, const Atom& atom);
  using Potential::compute;
  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial) override;

private:
  GPU_Exclusive_Scan cell_scan_;
  GPU_Vector<int> cell_count, cell_sum, cell_contents;
  int num_types;
  GPU_Vector<int> types;
  GPU_Vector<double> c6, c12;
  GPU_Vector<int> offsets, partners;
  GPU_Vector<double> lj_weights;
};
