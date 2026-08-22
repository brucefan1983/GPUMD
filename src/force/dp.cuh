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

#ifdef USE_DEEPMD
#pragma once
#include "DeepPot.h"
#include "neighbor.cuh"
#include "potential.cuh"
#include <stdio.h>
#include <vector>
#include <cstddef>

namespace deepmd_compat = deepmd;

// DP neighbor list, which is the same as lammps neighbor list
struct DP_NL {
  int inum;
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<int*> firstneigh;
  std::vector<int> neigh_storage;
};

enum {
  GHOST_X = 0,      // 0b001
  GHOST_Y = 1,      // 0b010
  GHOST_XY = 2,     // 0b011
  GHOST_Z = 3,      // 0b100
  GHOST_XZ = 4,     // 0b101
  GHOST_YZ = 5,     // 0b110
  GHOST_XYZ = 6     // 0b111
};

class DP : public Potential
{
public:
  using Potential::compute;
  DP(const char* , int);
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
  // dp coeff
  double ener_unit_cvt_factor;
  double dist_unit_cvt_factor;
  double force_unit_cvt_factor;
  double virial_unit_cvt_factor;
  bool atom_spin_flag;
  bool single_model;

  // neighbor-list skin cache
  double neighbor_skin;
  double neighbor_cutoff;
  bool nlist_cache_valid;
  bool deepmd_device_api_available;
  int cached_num_all_atoms;
  int cached_nghost;
  int cached_ndanger;
  int cached_number_of_atoms;
  int cached_pbc_x;
  int cached_pbc_y;
  int cached_pbc_z;
  double cached_box_h[9];

  // Device-resident neighbor list: the global list is built at rc + skin and
  // reused across steps until an atom drifts more than skin/2, then filtered
  // to the true cutoff rc each step.
  Neighbor dp_neighbor;
  GPU_Vector<int> dp_NN_local;   // per-step neighbor counts within rc
  GPU_Vector<int> dp_NL_local;   // per-step neighbor list within rc (stride N)
  DP_NL dp_nl;
  GPU_Vector<double> dp_position_gpu;
  std::vector<int> type_cpu;
  GPU_Vector<double> e_f_v_gpu;     // a temporary variable to save dp energy, force and virial

  // Atoms that need periodic DP image atoms are marked as dangerous.
  // Ghost atoms are translated by lattice-vector shifts, including triclinic cells.
  int nghost;                       // number of ghost atoms
  int ndanger;                      // number of dangerous atoms
  GPU_Vector<int> type_ghost;       // type of ghost atoms, nghost x 1
  GPU_Vector<int> ghost_count;      // count of ghost atoms for each local atom, number_of_atoms x 1
  GPU_Vector<int> ghost_sum;        // exclusive scan of ghost_count
  GPU_Vector<int> ghost_id_map;     // dangerous-atom ghost ids, ndanger x max_ghost_num_each_danger
  GPU_Vector<int> danger_flag;      // 1 if dangerous, 0 if not, number_of_atoms x 1
  GPU_Vector<int> danger_list;      // for each local atom: -1 or dense index in dangerous atom list
  GPU_Vector<int> danger_atoms;     // reverse map: dangerous atom list -> local atom index

  // dp instance
  deepmd_compat::DeepPot deep_pot;

  GPU_Vector<double> f_ghost;
  GPU_Vector<double> v_ghost;
  std::vector<int> cpu_NL;
  GPU_Vector<double> dp_position_gpu_trans;
  std::vector<double> dp_position_cpu;

  // Fully device-resident edge path (SeZM/DPA4): build the neighbor list and
  // the compact edge schema on the GPU, run the exported model on those device
  // tensors, and scatter the device outputs back into GPUMD arrays.  No host
  // neighbor-list build and no per-step host-device coordinate/result copies.
  GPU_Vector<int> dp_edge_index;     // [2 * nedge]: row 0 = src, row 1 = dst
  GPU_Vector<double> dp_edge_vec;    // [nedge * 3] minimum-image bond vectors
  GPU_Vector<int> dp_edge_offset;    // [nloc] exclusive scan of NN
  GPU_Vector<double> dp_atom_energy_gpu;  // [nloc]
  GPU_Vector<double> dp_force_rowmajor;   // [nloc * 3] row-major model force
  GPU_Vector<double> dp_atom_virial_gpu;  // [nloc * 9] row-major model virial
  void compute_gpu_edges(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  // skin-cache helper buffers
  GPU_Vector<double> position_ref_gpu;
  GPU_Vector<double> disp_sq_gpu;
  GPU_Vector<int> ghost_owner;
  GPU_Vector<int> ghost_shift_x;
  GPU_Vector<int> ghost_shift_y;
  GPU_Vector<int> ghost_shift_z;

  // auxiliary buffers for packed neighbor-list transfer
  GPU_Vector<int> nl_offset_gpu;
  GPU_Vector<int> nl_packed_gpu;
  std::vector<int> nl_offset_cpu;

  // dp output vectors
  std::vector<double> dp_ene_all;
  std::vector<double> dp_ene_atom;
  std::vector<double> dp_force;
  std::vector<double> dp_vir_all;
  std::vector<double> dp_vir_atom;

  // registered (page-locked) host buffer metadata for faster H2D/D2H copies
  void* type_cpu_pinned_ptr;
  size_t type_cpu_pinned_bytes;
  void* dp_position_cpu_pinned_ptr;
  size_t dp_position_cpu_pinned_bytes;
  void* cpu_NL_pinned_ptr;
  size_t cpu_NL_pinned_bytes;
  void* nl_offset_cpu_pinned_ptr;
  size_t nl_offset_cpu_pinned_bytes;
  void* dp_numneigh_pinned_ptr;
  size_t dp_numneigh_pinned_bytes;
  void* dp_ene_atom_pinned_ptr;
  size_t dp_ene_atom_pinned_bytes;
  void* dp_force_pinned_ptr;
  size_t dp_force_pinned_bytes;
  void* dp_vir_atom_pinned_ptr;
  size_t dp_vir_atom_pinned_bytes;

  void set_dp_coeff();
  void release_pinned_host_buffers();
};
#endif
