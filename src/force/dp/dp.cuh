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

#ifdef USE_DEEPMD
#pragma once
#include "DeepPot.h"
#include "force/neighbor.cuh"
#include "force/potential.cuh"
#include <cstddef>
#include <cstdint>
#include <stdio.h>
#include <vector>

namespace deepmd_compat = deepmd;

class DP : public Potential
{
public:
  using Potential::compute;
  DP(const char*, int);
  virtual ~DP(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
  void initialize_dp(const char* filename_dp);

protected:
  double ener_unit_cvt_factor;
  double dist_unit_cvt_factor;
  double force_unit_cvt_factor;
  double virial_unit_cvt_factor;

  // Device-resident neighbor list: the global list is built at rc + skin and
  // reused across steps until an atom drifts more than skin/2, then filtered
  // to the true cutoff rc each step.
  Neighbor dp_neighbor;
  GPU_Vector<int> dp_NN_local; // per-step neighbor counts within rc
  GPU_Vector<int> dp_NL_local; // per-step neighbor list within rc (stride N)
  GPU_Vector<double> dp_position_gpu;
  std::vector<int> type_cpu;
  GPU_Vector<double> e_f_v_gpu; // packed host-path output, or int64 source row pointer

  deepmd_compat::DeepPot deep_pot;

  // Canonical-graph scratch. Names match the values stored there.
  GPU_Vector<int> source_count;          // per-atom source counts
  GPU_Vector<int> source_cursor;         // exclusive source offsets, then write cursors
  GPU_Vector<std::int64_t> source_order; // source-grouped edge positions
  GPU_Vector<double> canonical_coord;    // row-major coordinates for a long-range module

  std::vector<double> dp_position_cpu;
  GPU_Vector<double> dp_position_gpu_trans;
  std::vector<double> fallback_box;

  // Fully device-resident edge path (SeZM/DPA4): build the neighbor list and
  // the compact edge schema on the GPU, run the exported model on those device
  // tensors, and scatter the device outputs back into GPUMD arrays.  No host
  // neighbor-list build and no per-step host-device coordinate/result copies.
  GPU_Vector<int> dp_edge_index;         // [2 * nedge]: row 0 = src, row 1 = dst
  GPU_Vector<double> dp_edge_vec;        // [nedge * 3] minimum-image bond vectors
  GPU_Vector<int> dp_edge_offset;        // [nloc] exclusive scan of NN
  GPU_Vector<double> dp_atom_energy_gpu; // [nloc]
  GPU_Vector<double> dp_force_rowmajor;  // [nloc * 3] row-major model force
  GPU_Vector<double> dp_atom_virial_gpu; // [nloc * 9] row-major model virial
  void compute_gpu_edges(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  std::vector<double> dp_ene_all;
  std::vector<double> dp_ene_atom;
  std::vector<double> dp_force;
  std::vector<double> dp_vir_all;
  std::vector<double> dp_vir_atom;

  void set_dp_coeff();
};
#endif
