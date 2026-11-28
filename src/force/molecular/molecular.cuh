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
struct MolecularAngle;

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
  GPU_Vector<int> cell_count_, cell_sum_, cell_contents_;
  int num_types_;
  GPU_Vector<int> types_;
  GPU_Vector<float> c6_, c12_;
  GPU_Vector<int> offsets_, partners_;
  GPU_Vector<float> lj_weights_;
  GPU_Vector<int> bond_offsets_, bond_partners_, bond_types_;
  GPU_Vector<float> bond_k_, bond_r0_;
  GPU_Vector<MolecularAngle> angles_;
  GPU_Vector<int> angle_offsets_, angle_indices_;
  GPU_Vector<float> angle_k_, angle_theta0_, angle_pi_minus_theta0_;
  GPU_Vector<int> angle_error_reported_;
};
