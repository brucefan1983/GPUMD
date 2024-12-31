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

/*----------------------------------------------------------------------------80
The class dealing with the Deep Potential(DP).
------------------------------------------------------------------------------*/


#include "dp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"





DP::DP(const char* filename_dp, int num_atoms)
{
  set_dp_coeff();


  dp_data.NN.resize(num_atoms);
  dp_data.NL.resize(num_atoms * 1024); // the largest supported by CUDA
  dp_data.cell_count.resize(num_atoms);
  dp_data.cell_count_sum.resize(num_atoms);
  dp_data.cell_contents.resize(num_atoms);
}

DP::~DP(void)
{
  // nothing
}

void DP::set_dp_coeff(void) {
  ener_unit_cvt_factor=1;      // 1.0 / 8.617343e-5;
  dist_unit_cvt_factor=1;      // 1;
  force_unit_cvt_factor=1;     // ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor=1;    // ener_unit_cvt_factor
  single_model = true;
  atom_spin_flag = false;
};



void DP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  return;
}