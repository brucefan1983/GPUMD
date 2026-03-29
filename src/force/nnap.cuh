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
#include "neighbor.cuh"
#include <vector>
#include <string>
#include <stdio.h>

#define MAX_NEIGH_NUM_NNAP 512

struct NNAP_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

// NNAP neighbor list in CPU format
struct NNAP_NL {
  int inum;
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<int*> firstneigh;
  std::vector<int> neigh_storage;
};

class NNAP : public Potential
{
public:
  using Potential::compute;
  NNAP(const char* filename, int num_atoms);
  virtual ~NNAP(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void initialize_nnap(const char* filename);

protected:
  double ener_unit_cvt_factor;
  double dist_unit_cvt_factor;
  double force_unit_cvt_factor;
  double virial_unit_cvt_factor;

  NNAP_Data nnap_data;
  NNAP_NL nnap_nl;

  GPU_Vector<double> position_gpu_trans;
  std::vector<double> position_cpu;
  std::vector<int> type_cpu;
  std::vector<double> box_cpu;
  std::vector<int> cpu_NL;

  // cache current GPUMD outputs (for future update)
  std::vector<double> gpumd_pe_cpu;
  std::vector<double> gpumd_force_cpu;
  std::vector<double> gpumd_virial_cpu;

  // temporary output buffers for future NNAP use
  std::vector<double> nnap_ene_atom;
  std::vector<double> nnap_force;
  std::vector<double> nnap_virial_atom;

  GPU_Vector<double> e_f_v_gpu; // reserved for future transpose/update

  void set_nnap_coeff();

  // empty API interface: NNAP developers only need to fill this
  void nnap_api_compute(
    int nlocal,
    const std::vector<double>& box,
    const std::vector<int>& type,
    const std::vector<double>& position,
    const NNAP_NL& nl,
    std::vector<double>& ene_atom,
    std::vector<double>& force,
    std::vector<double>& virial_atom);
};
