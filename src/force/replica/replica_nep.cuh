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
Energy NEP for replica dynamics.

The model (parameters, cutoffs, ZBL) and the kernels are those of NEP. What is
added is the state needed to evaluate several replicas on one GPU at the same
time: each stream has its own workspace (descriptors, neighbor lists, small-box
data), and a batch of replicas concatenated into one system can be evaluated
with one set of kernel launches.
------------------------------------------------------------------------------*/

#pragma once
#include "force/nep/nep.cuh"
#include "model/batch_box.cuh"
#include "replica_neighbor.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include <map>
#include <memory>
#include <mutex>

class NEP_Replica : public NEP
{
public:
  // num_atoms is the number of atoms evaluated per stream (all replicas of a batch);
  // verbose prints the model information.
  NEP_Replica(const char* file_potential, const int num_atoms, const bool verbose);

  using NEP::compute;

  // Allocates the workspace of a stream; compute() also does it on first use.
  void prepare_stream(gpuStream_t stream);

  // One replica on the workspace of the stream.
  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial,
    gpuStream_t stream);

  // The first number_of_atoms atoms as replicas concatenated into one system.
  void compute_batched(
    const Batch_Box& box,
    const int number_of_atoms,
    const int* type,
    const double* position,
    double* potential,
    double* force,
    double* virial,
    gpuStream_t stream);

  // Forces a neighbor-list rebuild, e.g. when a batch is filled with other replicas.
  void invalidate_neighbor_list(gpuStream_t stream);

  bool uses_small_box(const Box& box) const;

private:
  struct Workspace {
    NEP_Data data;
    Small_Box_Data small_box;
    ExpandedBox expanded_box;
    Replica_Neighbor neighbor;
    int number_of_calls = 0;
    GPU_Pinned_Vector<int> cpu_NN_radial;
    GPU_Pinned_Vector<int> cpu_NN_angular;
    std::mutex mutex; // one submission at a time per stream
  };

  int num_atoms_ = 0;
  int device_id_ = -1;
  std::map<gpuStream_t, std::unique_ptr<Workspace>> workspaces_;
  std::mutex workspaces_mutex_;
  std::mutex log_mutex_;

  Workspace& get_workspace(gpuStream_t stream);
  void report_neighbors(
    Workspace& workspace,
    GPU_Vector<int>& NN_radial,
    GPU_Vector<int>& NN_angular,
    const int number_of_atoms,
    const int number_of_replicas,
    gpuStream_t stream);
};
