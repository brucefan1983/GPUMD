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
Batched nvt_bdp integration of the replicas on one GPU.

The replica slots remain the owners of their state. For an MD block the active
slots are gathered into one concatenated system (replica i occupies atoms
[i * N, (i + 1) * N) of SoA arrays with stride i_max * N), advanced with one
kernel launch per stage and step, and scattered back. Each slot's thermostat
RNG is drawn in the same order as in the per-slot path, so the trajectories
are the same up to floating-point summation order.
------------------------------------------------------------------------------*/

#pragma once

#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include "utilities/gpu_stream.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <string>
#include <vector>

class NEP;

namespace replica
{

struct Replica_Slot;

struct Replica_Batch_Statistics {
  long long advances = 0;
  long long steps = 0;
  int longest_advance = 0;
  long long partial_advances = 0;       // fewer active slots than on the device
  long long noncontiguous_advances = 0; // active slots not one contiguous range
  long long full_to_partial = 0;
  long long partial_to_full = 0;
  long long noise_chunks = 0;
  long long noise_buffer_reuses = 0;    // chunks that waited to reuse a pinned buffer
  long long partial_tail_chunks = 0;    // last chunk of an advance shorter than a full chunk
};

class Replica_Batch
{
public:
  Replica_Batch(
    int device_id,
    int capacity,
    const std::string& potential_file,
    const Atom& source_atom,
    const Box& box);
  ~Replica_Batch();

  Replica_Batch(const Replica_Batch&) = delete;
  Replica_Batch& operator=(const Replica_Batch&) = delete;

  // False when the box would need GPUMD's small-box NEP path.
  bool supported() const { return supported_; }

  // Advance the given slots (all on this device) by steps nvt_bdp steps and
  // update their positions, velocities, forces, potentials, virials, thermo
  // and host_thermo / potential_energy, as the per-slot path does.
  void advance(const std::vector<Replica_Slot*>& slots, int steps, double time_step);

  // Called by the runtime before advance() with the shape of the active selection.
  void note_selection(bool full, bool contiguous);
  const Replica_Batch_Statistics& statistics() const { return statistics_; }

private:
  static constexpr int chunk_steps_ = 256;

  int device_id_;
  int capacity_;
  int atoms_per_replica_;
  bool supported_ = false;
  bool previous_full_ = true;
  Replica_Batch_Statistics statistics_;
  Box box_;
  std::unique_ptr<NEP> nep_;
  GPU_Stream stream_;
  GPU_Event chunk_done_[2];

  GPU_Vector<int> type_;
  GPU_Vector<double> mass_;
  GPU_Vector<double> position_;
  GPU_Vector<double> velocity_;
  GPU_Vector<double> force_;
  GPU_Vector<double> potential_;
  GPU_Vector<double> virial_;
  GPU_Vector<double> thermo_;
  GPU_Vector<double> target_temperature_;
  GPU_Vector<double> noise_[2];
  GPU_Pinned_Vector<double> host_noise_[2];
  GPU_Pinned_Vector<double> host_target_temperature_;
  GPU_Pinned_Vector<double> host_thermo_;

  void gather(const std::vector<Replica_Slot*>& slots, int stride);
  void scatter(const std::vector<Replica_Slot*>& slots, int stride);
};

} // namespace replica
