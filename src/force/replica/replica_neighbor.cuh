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
Global (candidate) neighbor list for replica dynamics.

Unlike the Neighbor class, every operation runs on a given stream, so that
replicas sharing a GPU can be evaluated concurrently, and a list can cover
replicas concatenated into one system (Batch_Box), with one set of cells per
replica so that atoms only see atoms of their own replica.

The list is rebuilt when an atom has moved by more than half the skin since
the last build. The check runs on the device and the rebuild kernels return
early when it is not needed, so no host synchronization is involved. Lists are
sorted, so a list built at another step gives the same NEP neighbor lists.
------------------------------------------------------------------------------*/

#pragma once
#include "model/batch_box.cuh"
#include "model/box.cuh"
#include "utilities/gpu_scan.cuh"
#include "utilities/gpu_vector.cuh"

class Replica_Neighbor
{
public:
  GPU_Vector<int> NN, NL; // global neighbor list

  // num_atoms is the capacity, e.g. all replicas of a batch.
  void initialize(const double rc, const int num_atoms, const int num_neighbors);

  // Forces a rebuild at the next update, e.g. when a batch holds other replicas.
  void invalidate() { valid_ = false; }

  void update(
    const double rc,
    const Box& box,
    const int number_of_atoms,
    const double* position_per_atom,
    gpuStream_t stream);

  void update(
    const double rc,
    const Batch_Box& box,
    const int number_of_atoms,
    const double* position_per_atom,
    gpuStream_t stream);

private:
  double skin_ = 1.0;
  bool valid_ = false;
  int built_number_of_atoms_ = 0;
  GPU_Vector<int> cell_count_;
  GPU_Vector<int> cell_count_sum_;
  GPU_Vector<int> cell_contents_;
  GPU_Vector<double> position_at_build_;
  GPU_Vector<int> rebuild_; // one flag on the device
  GPU_Exclusive_Scan scan_;

  // cell_box is the (shared) box inside box, whose thickness get_num_bins() sets for
  // the kernels; both are references so that the kernels receive the updated box.
  template <typename Box_View>
  void update_list(
    const double rc,
    Box_View& box,
    Box& cell_box,
    const int number_of_replicas,
    const int number_of_atoms,
    const double* position_per_atom,
    gpuStream_t stream);
};
