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
#include "DeepPot.h"
#include "potential.cuh"
#include <stdio.h>
#include <vector>

namespace deepmd_compat = deepmd;


struct DP_Data {
  GPU_Vector<int> NN, NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
};

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

  DP_Data dp_data;
  DP_NL dp_nl;
  GPU_Vector<double> dp_position_gpu;
  std::vector<int> type_cpu;
  GPU_Vector<double> e_f_v_gpu;     // a temporary variable to save dp energy, force and virial

  // the atoms whose distance to boundaries is less than rcut are dangerous
  // the atoms mirrored by dangerous atoms are ghost atoms
  int nghost;                       // number of ghost atoms
  int ndanger;                      // number of dangerous atoms
  GPU_Vector<int> type_ghost;       // type of ghost atoms, nghost x 1
  GPU_Vector<int> ghost_count;      // count of ghost atoms for each local atom, number_of_atoms x 1
  GPU_Vector<int> ghost_sum;        // exclusive the ghost_count
  GPU_Vector<int> nghost_tmp;       // a temporary vector to save ghost atom number
  GPU_Vector<int> ghost_id_map;     // a map to find the ghost id of each dangerous atom, ndanger x 7
  GPU_Vector<int> danger_flag;      // 1 if dangerous, 0 if not, number_of_atoms x 1
  GPU_Vector<int> danger_list;      // the dangerous atom index list, according to ghost_id_map, number_of_atoms x 1

  // these vectors are used to get the neighbors of the ghost atoms
  GPU_Vector<int> ghost_numneigh;   // save the neighbor number of ghost atoms, nghost x 1
  GPU_Vector<int> ghost_numneigh_sum; // exclusive the ghost_numneigh
  GPU_Vector<int> ghost_neigh_pair; // each pair save (ghost id, neigh), nghost x max_neigh x 2

  // dp instance
  deepmd_compat::DeepPot deep_pot;

  void set_dp_coeff();
};
