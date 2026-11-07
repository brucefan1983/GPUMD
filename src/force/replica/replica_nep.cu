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
Energy NEP for replica dynamics, see replica_nep.cuh.
------------------------------------------------------------------------------*/

#include "replica_nep.cuh"
#include "replica_nep_kernels.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <fstream>

NEP_Replica::NEP_Replica(const char* file_potential, const int num_atoms, const bool verbose)
  : NEP_Replica_Model(file_potential, verbose), num_atoms_(num_atoms)
{
  CHECK(gpuGetDevice(&device_id_));
}

NEP_Replica::Workspace& NEP_Replica::get_workspace(const gpuStream_t stream)
{
  if (stream == nullptr)
    PRINT_INPUT_ERROR("Replica NEP evaluation requires an explicit GPU stream.\n");
  int current_device = -1;
  CHECK(gpuGetDevice(&current_device));
  if (current_device != device_id_)
    PRINT_INPUT_ERROR("A replica NEP is used on another GPU than the one it was built on.\n");

  std::lock_guard<std::mutex> lock(workspaces_mutex_);
  auto iterator = workspaces_.find(stream);
  if (iterator != workspaces_.end())
    return *iterator->second;

  std::unique_ptr<Workspace> workspace(new Workspace);
  NEP_Replica_Data& data = workspace->data;
  const size_t N = num_atoms_;
  data.f12x.resize(N * paramb.MN_angular);
  data.f12y.resize(N * paramb.MN_angular);
  data.f12z.resize(N * paramb.MN_angular);
  data.NN_radial.resize(N);
  data.NL_radial.resize(N * paramb.MN_radial);
  data.NN_angular.resize(N);
  data.NL_angular.resize(N * paramb.MN_angular);
  data.Fp.resize(N * annmb.dim);
  data.sum_fxyz.resize(
    N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  workspace->neighbor.initialize(rc, num_atoms_, paramb.MN_radial);
  workspace->cpu_NN_radial.resize(N);
  workspace->cpu_NN_angular.resize(N);
  return *workspaces_.emplace(stream, std::move(workspace)).first->second;
}

void NEP_Replica::prepare_stream(const gpuStream_t stream) { get_workspace(stream); }

void NEP_Replica::invalidate_neighbor_list(const gpuStream_t stream)
{
  Workspace& workspace = get_workspace(stream);
  std::lock_guard<std::mutex> lock(workspace.mutex);
  workspace.neighbor.invalidate();
}

bool NEP_Replica::uses_small_box(const Box& box) const
{
  ExpandedBox unused;
  Box copy = box;
  copy.set_is_orthogonal();
  return get_expanded_box(paramb.rc_radial_max, copy, unused);
}

// Appends the largest neighbor counts to neighbor.out every 1000 calls of a workspace.
void NEP_Replica::report_neighbors(
  Workspace& workspace,
  GPU_Vector<int>& NN_radial,
  GPU_Vector<int>& NN_angular,
  const int number_of_atoms,
  const int number_of_replicas,
  const gpuStream_t stream)
{
  const int call_index = workspace.number_of_calls++;
  if (call_index % 1000 != 0)
    return;
  NN_radial.copy_to_host_async(workspace.cpu_NN_radial.data(), number_of_atoms, stream);
  NN_angular.copy_to_host_async(workspace.cpu_NN_angular.data(), number_of_atoms, stream);
  CHECK(gpuStreamSynchronize(stream));
  int radial_actual = 0;
  int angular_actual = 0;
  for (int n = 0; n < number_of_atoms; ++n) {
    radial_actual = std::max(radial_actual, workspace.cpu_NN_radial[n]);
    angular_actual = std::max(angular_actual, workspace.cpu_NN_angular[n]);
  }
  std::lock_guard<std::mutex> lock(log_mutex_);
  std::ofstream output("neighbor.out", std::ios_base::app);
  output << "Neighbor info at step " << call_index;
  if (number_of_replicas > 1)
    output << " (batch of " << number_of_replicas << " replicas)";
  output << ": radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
         << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
         << std::endl;
}

void NEP_Replica::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  const gpuStream_t stream)
{
  const int N = type.size();
  if (N != num_atoms_)
    PRINT_INPUT_ERROR("The replica NEP workspace atom count cannot change.\n");
  Workspace& workspace = get_workspace(stream);
  std::lock_guard<std::mutex> lock(workspace.mutex);
  Box stream_box = box;
  stream_box.set_is_orthogonal();

  if (get_expanded_box(paramb.rc_radial_max, stream_box, workspace.expanded_box)) {
    Small_Box_Data& small_box = workspace.small_box;
    if (small_box.NN_radial.size() != N) {
      const int big_neighbor_size = 2000;
      const size_t size_x12 = static_cast<size_t>(N) * big_neighbor_size;
      small_box.NN_radial.resize(N);
      small_box.NL_radial.resize(size_x12);
      small_box.NN_angular.resize(N);
      small_box.NL_angular.resize(size_x12);
      small_box.r12.resize(size_x12 * 6);
    }
    find_neighbor_lists_small_box(
      stream_box, workspace.expanded_box, N, 0, N, type.data(), position.data(), small_box,
      stream);
    report_neighbors(workspace, small_box.NN_radial, small_box.NN_angular, N, 1, stream);
    compute_small_box_from_neighbor_lists(
      N, 0, N, type.data(), potential.data(), force.data(), virial.data(), workspace.data,
      small_box, stream);
    return;
  }

  workspace.neighbor.update(rc, stream_box, N, position.data(), stream);
  find_neighbor_lists_large_box(
    stream_box, N, 0, N, type.data(), position.data(), workspace.neighbor.NN.data(),
    workspace.neighbor.NL.data(), workspace.data, stream);
  report_neighbors(workspace, workspace.data.NN_radial, workspace.data.NN_angular, N, 1, stream);
  compute_large_box_from_neighbor_lists(
    stream_box, N, 0, N, type.data(), position.data(), potential.data(), force.data(),
    virial.data(), workspace.data, stream);
}

void NEP_Replica::compute_batched(
  const Batch_Box& box,
  const int number_of_atoms,
  const int* type,
  const double* position,
  double* potential,
  double* force,
  double* virial,
  const gpuStream_t stream)
{
  const int N = number_of_atoms;
  if (N <= 0 || N > num_atoms_ || box.atoms_per_replica <= 0 || N % box.atoms_per_replica != 0)
    PRINT_INPUT_ERROR("The batched NEP workspace does not match the batch.\n");
  Batch_Box batch_box = box;
  batch_box.box.set_is_orthogonal();
  if (uses_small_box(batch_box.box))
    PRINT_INPUT_ERROR("Batched NEP evaluation does not support small boxes.\n");

  Workspace& workspace = get_workspace(stream);
  std::lock_guard<std::mutex> lock(workspace.mutex);
  workspace.neighbor.update(rc, batch_box, N, position, stream);
  find_neighbor_lists_large_box(
    batch_box, N, 0, N, type, position, workspace.neighbor.NN.data(),
    workspace.neighbor.NL.data(), workspace.data, stream);
  report_neighbors(
    workspace, workspace.data.NN_radial, workspace.data.NN_angular, N,
    N / box.atoms_per_replica, stream);
  compute_large_box_from_neighbor_lists(
    batch_box, N, 0, N, type, position, potential, force, virial, workspace.data, stream);
}
