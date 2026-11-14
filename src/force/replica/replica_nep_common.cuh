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

/*----------------------------------------------------------------------------80
Pieces shared by the replica NEP and qNEP models (replica_nep_model.cuh and
replica_nep_charge_model.cuh): the small-box data, the expanded box, element
lookup and the neighbor.out report.
------------------------------------------------------------------------------*/

#pragma once

#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>

// Room for this many neighbors per atom in the small-box neighbor lists.
constexpr int REPLICA_SMALL_BOX_NEIGHBORS = 2000;

struct Replica_Expanded_Box {
  int num_cells[3];
  float h[18];
};

struct Replica_Small_Box_Data {
  GPU_Vector<int> NN_radial;
  GPU_Vector<int> NL_radial;
  GPU_Vector<int> NN_angular;
  GPU_Vector<int> NL_angular;
  GPU_Vector<float> r12;

  // Resizes for N atoms unless already sized for them.
  void resize(int N);
};

// Whether the box needs the small-box path; also fills the expanded box for it.
bool get_expanded_box(double rc, const Box& box, Replica_Expanded_Box& ebox);

// The atomic number of an element symbol, or 0 if it is unknown.
int atomic_number(const std::string& element);

// Appends the largest of the N neighbor counts to neighbor.out.
void write_neighbor_info(
  int call_index,
  int replicas,
  int MN_radial,
  int MN_angular,
  const int* NN_radial,
  const int* NN_angular,
  int N);
