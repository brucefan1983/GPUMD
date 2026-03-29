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

#ifdef USE_NNAP
#include "nnap.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>
#include <iostream>

#define BLOCK_SIZE_FORCE 128

NNAP::NNAP(const char* filename, int num_atoms)
{
  set_nnap_coeff();
  initialize_nnap(filename);

  nnap_data.NN.resize(num_atoms);
  nnap_data.NL.resize(num_atoms * MAX_NEIGH_NUM_NNAP); // should define MAX_NEIGH_NUM_NNAP from the NNAP.txt potential file.
  nnap_data.cell_count.resize(num_atoms);
  nnap_data.cell_count_sum.resize(num_atoms);
  nnap_data.cell_contents.resize(num_atoms);

  position_gpu_trans.resize(num_atoms * 3);
  type_cpu.resize(num_atoms);
  e_f_v_gpu.resize(num_atoms * (1 + 3 + 9));

  nnap_nl.inum = num_atoms;
  nnap_nl.ilist.resize(num_atoms);
  nnap_nl.numneigh.resize(num_atoms);
  nnap_nl.firstneigh.resize(num_atoms);
  nnap_nl.neigh_storage.resize(num_atoms * MAX_NEIGH_NUM_NNAP);
}

NNAP::~NNAP(void)
{
  // empty
}

void NNAP::initialize_nnap(const char* filename)
{
  printf("\nInitialize empty NNAP driver by file: %s\n\n", filename);
  rc = 6.0; // placeholder cutoff, NNAP developers can modify
}

void NNAP::set_nnap_coeff(void)
{
  ener_unit_cvt_factor = 1.0;
  dist_unit_cvt_factor = 1.0;
  force_unit_cvt_factor = ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor = 1.0;
}

static __global__ void position_transpose(
  const double* position,
  double* position_trans,
  int N)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    position_trans[n * 3 + 0] = position[n];
    position_trans[n * 3 + 1] = position[n + N];
    position_trans[n * 3 + 2] = position[n + 2 * N];
  }
}

void NNAP::nnap_api_compute(
  int nlocal,
  const std::vector<double>& box,
  const std::vector<int>& type,
  const std::vector<double>& position,
  const NNAP_NL& nl,
  std::vector<double>& ene_atom,
  std::vector<double>& force,
  std::vector<double>& virial_atom)
{
  // ================================
  // Empty NNAP API interface
  // NNAP developers only need to fill here:
  //
  // input:
  //   nlocal              : number of local atoms
  //   box[9]              : box matrix
  //   type[nlocal]        : atom types
  //   position[nlocal*3]  : xyzxyz...
  //   nl                  : CPU neighbor list
  //
  // output format:
  //   ene_atom[nlocal]
  //   force[nlocal*3]       : fx fy fz ...
  //   virial_atom[nlocal*9] : 3x3 per atom
  //
  // current template does nothing
  // ================================
}

void NNAP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  const int number_of_atoms = type.size();
  const int grid_size = (number_of_atoms - 1) / BLOCK_SIZE_FORCE + 1;

  // 1) build GPUMD neighbor list
  nnap_data.NN.resize(number_of_atoms);
  nnap_data.NL.resize(number_of_atoms * MAX_NEIGH_NUM_NNAP);
  nnap_data.cell_count.resize(number_of_atoms);
  nnap_data.cell_count_sum.resize(number_of_atoms);
  nnap_data.cell_contents.resize(number_of_atoms);

  find_neighbor(
    N1,
    N2,
    rc,
    box,
    type,
    position,
    nnap_data.cell_count,
    nnap_data.cell_count_sum,
    nnap_data.cell_contents,
    nnap_data.NN,
    nnap_data.NL);

  // 2) copy type
  type_cpu.resize(number_of_atoms);
  type.copy_to_host(type_cpu.data());

  // 3) copy position and transpose to xyzxyz...
  position_gpu_trans.resize(number_of_atoms * 3);
  position_transpose<<<grid_size, BLOCK_SIZE_FORCE>>>(
    position.data(),
    position_gpu_trans.data(),
    number_of_atoms);
  GPU_CHECK_KERNEL

  position_cpu.resize(number_of_atoms * 3);
  position_gpu_trans.copy_to_host(position_cpu.data());

  // 4) copy box to CPU (3x3)
  box_cpu.assign(9, 0.0);
  box_cpu[0] = box.cpu_h[0];
  box_cpu[1] = box.cpu_h[1];
  box_cpu[2] = box.cpu_h[2];
  box_cpu[3] = box.cpu_h[3];
  box_cpu[4] = box.cpu_h[4];
  box_cpu[5] = box.cpu_h[5];
  box_cpu[6] = box.cpu_h[6];
  box_cpu[7] = box.cpu_h[7];
  box_cpu[8] = box.cpu_h[8];

  // 5) copy current GPUMD outputs for future update
  gpumd_pe_cpu.resize(number_of_atoms);
  gpumd_force_cpu.resize(number_of_atoms * 3);
  gpumd_virial_cpu.resize(number_of_atoms * 9);

  potential.copy_to_host(gpumd_pe_cpu.data());
  force.copy_to_host(gpumd_force_cpu.data());
  virial.copy_to_host(gpumd_virial_cpu.data());

  // 6) convert GPUMD neighbor list to CPU pointer-style list
  nnap_nl.inum = number_of_atoms;
  nnap_nl.ilist.resize(number_of_atoms);
  nnap_nl.numneigh.resize(number_of_atoms);
  nnap_nl.firstneigh.resize(number_of_atoms);

  nnap_data.NN.copy_to_host(nnap_nl.numneigh.data());
  cpu_NL.resize(nnap_data.NL.size());
  nnap_data.NL.copy_to_host(cpu_NL.data());

  int offset = 0;
  nnap_nl.neigh_storage.resize(cpu_NL.size());
  for (int i = 0; i < number_of_atoms; ++i) {
    nnap_nl.ilist[i] = i;
    nnap_nl.firstneigh[i] = nnap_nl.neigh_storage.data() + offset;
    for (int j = 0; j < nnap_nl.numneigh[i]; ++j) {
      nnap_nl.neigh_storage[offset + j] = cpu_NL[i + j * number_of_atoms];
    }
    offset += nnap_nl.numneigh[i];
  }

  // 7) reserve NNAP output buffers
  nnap_ene_atom.assign(number_of_atoms, 0.0);
  nnap_force.assign(number_of_atoms * 3, 0.0);
  nnap_virial_atom.assign(number_of_atoms * 9, 0.0);

  // 8) empty NNAP API call
  nnap_api_compute(
    number_of_atoms,
    box_cpu,
    type_cpu,
    position_cpu,
    nnap_nl,
    nnap_ene_atom,
    nnap_force,
    nnap_virial_atom);

  // 9) update GPUMD outputs
  for (int i = 0; i < number_of_atoms; ++i) {
    gpumd_pe_cpu[i] += nnap_ene_atom[i] * ener_unit_cvt_factor;
  }
  for (int i = 0; i < number_of_atoms * 3; ++i) {
    gpumd_force_cpu[i] += nnap_force[i] * force_unit_cvt_factor;
  }
  for (int i = 0; i < number_of_atoms * 9; ++i) {
    gpumd_virial_cpu[i] += nnap_virial_atom[i] * virial_unit_cvt_factor;
  }
  
  potential.copy_from_host(gpumd_pe_cpu.data());
  force.copy_from_host(gpumd_force_cpu.data());
  virial.copy_from_host(gpumd_virial_cpu.data());
  
}
#endif
