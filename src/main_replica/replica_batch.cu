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

#include "replica_batch.cuh"
#include "replica_common.cuh"
#include "force/nep.cuh"
#include "model/batch_box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>

namespace
{

// Same arithmetic as gpu_velocity_verlet in integrate/ensemble.cu.
__global__ void gpu_velocity_verlet_batched(
  const bool is_step1,
  const int number_of_particles,
  const double g_time_step,
  const double* g_mass,
  double* g_x,
  double* g_y,
  double* g_z,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double time_step = g_time_step;
    const double time_step_half = time_step * 0.5;
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    const double mass_inv = 1.0 / g_mass[i];
    const double ax = g_fx[i] * mass_inv;
    const double ay = g_fy[i] * mass_inv;
    const double az = g_fz[i] * mass_inv;
    vx += ax * time_step_half;
    vy += ay * time_step_half;
    vz += az * time_step_half;
    g_vx[i] = vx;
    g_vy[i] = vy;
    g_vz[i] = vz;
    if (is_step1) {
      g_x[i] += vx * time_step;
      g_y[i] += vy * time_step;
      g_z[i] += vz * time_step;
    }
  }
}

// Same as apply_pbc followed by initialize_properties in force/force.cu.
__global__ void gpu_prepare_force_batched(
  const int number_of_atoms,
  const Box box,
  double* x,
  double* y,
  double* z,
  double* force_x,
  double* force_y,
  double* force_z,
  double* potential,
  double* virial)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= number_of_atoms)
    return;

  double sx = box.cpu_h[9] * x[n] + box.cpu_h[10] * y[n] + box.cpu_h[11] * z[n];
  double sy = box.cpu_h[12] * x[n] + box.cpu_h[13] * y[n] + box.cpu_h[14] * z[n];
  double sz = box.cpu_h[15] * x[n] + box.cpu_h[16] * y[n] + box.cpu_h[17] * z[n];
  // Rewrite the coordinates only when the atom is wrapped: the x -> s -> x round
  // trip is not exact, so rewriting every atom would perturb positions at the last
  // bit each time (e.g. an extra force evaluation after a resume).
  bool wrapped = false;
  if (box.pbc_x == 1) {
    if (sx < 0.0) {
      sx += 1.0;
      wrapped = true;
    } else if (sx > 1.0) {
      sx -= 1.0;
      wrapped = true;
    }
  }
  if (box.pbc_y == 1) {
    if (sy < 0.0) {
      sy += 1.0;
      wrapped = true;
    } else if (sy > 1.0) {
      sy -= 1.0;
      wrapped = true;
    }
  }
  if (box.pbc_z == 1) {
    if (sz < 0.0) {
      sz += 1.0;
      wrapped = true;
    } else if (sz > 1.0) {
      sz -= 1.0;
      wrapped = true;
    }
  }
  if (wrapped) {
    x[n] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
    y[n] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
    z[n] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
  }

  force_x[n] = 0.0;
  force_y[n] = 0.0;
  force_z[n] = 0.0;
  potential[n] = 0.0;
  for (int component = 0; component < 9; ++component)
    virial[n + component * number_of_atoms] = 0.0;
}

// Batched copy of gpu_find_thermo_instant_temperature (integrate/ensemble.cu):
// g_thermo[r * 8 + 0..7] = T, U, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz of replica r.
static __global__ void gpu_find_thermo_batched(
  const int N,
  const int degrees_of_freedom,
  const double T,
  const double volume,
  const double* g_mass,
  const double* g_potential,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_sxx,
  const double* g_syy,
  const double* g_szz,
  const double* g_sxy,
  const double* g_sxz,
  const double* g_syz,
  double* g_thermo)
{
  //<<<dim3(8, replicas), 1024>>>: blockIdx.x = quantity, blockIdx.y = replica
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  const int replica_offset = blockIdx.y * N;
  int patch, n;
  int number_of_patches = (N - 1) / 1024 + 1;
  double mass, vx, vy, vz;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  switch (bid) {
    // temperature
    case 0:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vx = g_vx[replica_offset + n];
          vy = g_vy[replica_offset + n];
          vz = g_vz[replica_offset + n];
          s_data[tid] += (vx * vx + vy * vy + vz * vz) * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 0] = s_data[0] / (degrees_of_freedom * K_B);
      }
      break;
      // potential energy
    case 1:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          s_data[tid] += g_potential[replica_offset + n];
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0)
        g_thermo[blockIdx.y * 8 + 1] = s_data[0];
      break;
      // sxx
    case 2:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vx = g_vx[replica_offset + n];
          s_data[tid] += g_sxx[replica_offset + n] + vx * vx * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 2] = s_data[0] / volume;
      }
      break;
      // syy
    case 3:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vy = g_vy[replica_offset + n];
          s_data[tid] += g_syy[replica_offset + n] + vy * vy * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 3] = s_data[0] / volume;
      }
      break;
      // szz
    case 4:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vz = g_vz[replica_offset + n];
          s_data[tid] += g_szz[replica_offset + n] + vz * vz * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 4] = s_data[0] / volume;
      }
      break;
      // sxy
    case 5:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vx = g_vx[replica_offset + n];
          vy = g_vy[replica_offset + n];
          s_data[tid] += g_sxy[replica_offset + n] + vx * vy * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 5] = s_data[0] / volume;
      }
      break;
      // sxz
    case 6:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vx = g_vx[replica_offset + n];
          vz = g_vz[replica_offset + n];
          s_data[tid] += g_sxz[replica_offset + n] + vx * vz * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 6] = s_data[0] / volume;
      }
      break;
      // syz
    case 7:
      for (patch = 0; patch < number_of_patches; ++patch) {
        n = tid + patch * 1024;
        if (n < N) {
          mass = g_mass[replica_offset + n];
          vz = g_vz[replica_offset + n];
          vy = g_vy[replica_offset + n];
          s_data[tid] += g_syz[replica_offset + n] + vy * vz * mass;
        }
      }
      __syncthreads();
      for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
          s_data[tid] += s_data[tid + offset];
        }
        __syncthreads();
      }
      if (tid == 0) {
        g_thermo[blockIdx.y * 8 + 7] = s_data[0] / volume;
      }
      break;
  }
}

// Same arithmetic as gpu_resamplekin_and_scale in integrate/ensemble_bdp.cu, with
// per-replica target temperature, noise and thermo.
__global__ void gpu_resamplekin_and_scale_batched(
  const int number_of_atoms,
  const int atoms_per_replica,
  const int ndeg,
  const double temperature_coupling,
  const double* target_temperature,
  const double* noise,
  const double* thermo,
  double* velocity_x,
  double* velocity_y,
  double* velocity_z)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= number_of_atoms)
    return;
  const int replica = atom / atoms_per_replica;
  const double temperature = target_temperature[replica];
  const double gaussian = noise[replica * 2];
  const double sum_of_squares = noise[replica * 2 + 1];

  const double kinetic_energy = thermo[replica * 8] * ndeg * K_B * 0.5;
  const double target_kinetic_energy = ndeg * K_B * temperature * 0.5;
  const double coupling =
    temperature_coupling > 0.1 ? exp(-1.0 / temperature_coupling) : 0.0;
  const double new_kinetic_energy =
    kinetic_energy +
    (1.0 - coupling) *
      (target_kinetic_energy * (sum_of_squares + gaussian * gaussian) / ndeg -
       kinetic_energy) +
    2.0 * gaussian *
      sqrt(
        kinetic_energy * target_kinetic_energy / ndeg * (1.0 - coupling) * coupling);
  const double factor = sqrt(new_kinetic_energy / kinetic_energy);
  velocity_x[atom] *= factor;
  velocity_y[atom] *= factor;
  velocity_z[atom] *= factor;
}

// Copy `components` blocks of `count` doubles between arrays whose component
// strides differ (one replica's arrays and the concatenated batch arrays).
void copy_components(
  double* destination,
  const int destination_stride,
  const double* source,
  const int source_stride,
  const int count,
  const int components,
  const gpuStream_t stream)
{
  CHECK(gpuMemcpy2DAsync(
    destination,
    sizeof(double) * destination_stride,
    source,
    sizeof(double) * source_stride,
    sizeof(double) * count,
    components,
    gpuMemcpyDeviceToDevice,
    stream));
}

} // namespace

namespace replica
{

Replica_Batch::Replica_Batch(
  const int device_id,
  const int capacity,
  const std::string& potential_file,
  const Atom& source_atom,
  const Box& box)
  : device_id_(device_id)
  , capacity_(capacity)
  , atoms_per_replica_(source_atom.number_of_atoms)
  , box_(box)
{
  if (capacity_ <= 0 || atoms_per_replica_ <= 0)
    PRINT_INPUT_ERROR("Invalid replica batch size.");
  gpu_detail::Device_Guard guard(device_id_);
  box_.set_is_orthogonal();
  const int total = capacity_ * atoms_per_replica_;
  nep_.reset(new NEP(potential_file.c_str(), total, false));
  if (nep_->nep_model_type != 0)
    PRINT_INPUT_ERROR("Batched replicas only support energy NEP models.");
  nep_->N1 = 0;
  nep_->N2 = total;
  supported_ = !nep_->uses_small_box(box_);
  if (!supported_)
    return;

  stream_.create(device_id_);
  chunk_done_[0].create(device_id_);
  chunk_done_[1].create(device_id_);
  nep_->prepare_stream(stream_.get());

  std::vector<int> type(total);
  std::vector<double> mass(total);
  for (int replica = 0; replica < capacity_; ++replica) {
    for (int atom = 0; atom < atoms_per_replica_; ++atom) {
      type[replica * atoms_per_replica_ + atom] = source_atom.cpu_type[atom];
      mass[replica * atoms_per_replica_ + atom] = source_atom.cpu_mass[atom];
    }
  }
  type_.resize(total);
  type_.copy_from_host(type.data());
  mass_.resize(total);
  mass_.copy_from_host(mass.data());
  position_.resize(static_cast<size_t>(total) * 3);
  velocity_.resize(static_cast<size_t>(total) * 3);
  force_.resize(static_cast<size_t>(total) * 3);
  potential_.resize(total);
  virial_.resize(static_cast<size_t>(total) * 9);
  thermo_.resize(static_cast<size_t>(capacity_) * 8);
  target_temperature_.resize(capacity_);
  host_target_temperature_.resize(capacity_);
  host_thermo_.resize(static_cast<size_t>(capacity_) * 8);
  for (int b = 0; b < 2; ++b) {
    noise_[b].resize(static_cast<size_t>(chunk_steps_) * capacity_ * 2);
    host_noise_[b].resize(static_cast<size_t>(chunk_steps_) * capacity_ * 2);
  }
}

Replica_Batch::~Replica_Batch()
{
  CHECK(gpuSetDevice(device_id_));
  if (stream_.valid())
    stream_.synchronize();
}

void Replica_Batch::note_selection(const bool full, const bool contiguous)
{
  if (!full) {
    ++statistics_.partial_advances;
    if (!contiguous)
      ++statistics_.noncontiguous_advances;
  }
  if (previous_full_ && !full)
    ++statistics_.full_to_partial;
  if (!previous_full_ && full)
    ++statistics_.partial_to_full;
  previous_full_ = full;
}

void Replica_Batch::gather(const std::vector<Replica_Slot*>& slots, const int stride)
{
  const int N = atoms_per_replica_;
  for (size_t i = 0; i < slots.size(); ++i) {
    Atom& atom = slots[i]->atom;
    copy_components(position_.data() + i * N, stride, atom.position_per_atom.data(), N, N, 3, stream_.get());
    copy_components(velocity_.data() + i * N, stride, atom.velocity_per_atom.data(), N, N, 3, stream_.get());
    copy_components(force_.data() + i * N, stride, atom.force_per_atom.data(), N, N, 3, stream_.get());
  }
}

void Replica_Batch::scatter(const std::vector<Replica_Slot*>& slots, const int stride)
{
  const int N = atoms_per_replica_;
  for (size_t i = 0; i < slots.size(); ++i) {
    Replica_Slot& slot = *slots[i];
    Atom& atom = slot.atom;
    copy_components(atom.position_per_atom.data(), N, position_.data() + i * N, stride, N, 3, stream_.get());
    copy_components(atom.velocity_per_atom.data(), N, velocity_.data() + i * N, stride, N, 3, stream_.get());
    copy_components(atom.force_per_atom.data(), N, force_.data() + i * N, stride, N, 3, stream_.get());
    copy_components(atom.potential_per_atom.data(), N, potential_.data() + i * N, stride, N, 1, stream_.get());
    copy_components(atom.virial_per_atom.data(), N, virial_.data() + i * N, stride, N, 9, stream_.get());
    CHECK(gpuMemcpyAsync(
      slot.thermo.data(), thermo_.data() + i * 8, sizeof(double) * 8,
      gpuMemcpyDeviceToDevice, stream_.get()));
  }
}

void Replica_Batch::advance(
  const std::vector<Replica_Slot*>& slots,
  const int steps,
  const double time_step)
{
  if (!supported_)
    PRINT_INPUT_ERROR("Internal error: unsupported replica batch was used.");
  if (slots.empty() || steps <= 0)
    return;
  const int replicas = static_cast<int>(slots.size());
  if (replicas > capacity_)
    PRINT_INPUT_ERROR("Internal error: replica batch capacity exceeded.");
  CHECK(gpuSetDevice(device_id_));

  ++statistics_.advances;
  statistics_.steps += steps;
  statistics_.longest_advance = std::max(statistics_.longest_advance, steps);

  const int N = atoms_per_replica_;
  const int total = replicas * N;
  const Ensemble_BDP& first = *slots[0]->thermostat;
  const int ndeg = 3 * N - first.removed_degrees_of_freedom;
  const double coupling = first.temperature_coupling;
  for (int i = 0; i < replicas; ++i) {
    const Replica_Slot& slot = *slots[i];
    if (slot.device_id != device_id_ || !slot.thermostat ||
        slot.thermostat->removed_degrees_of_freedom != first.removed_degrees_of_freedom ||
        slot.thermostat->temperature_coupling != coupling ||
        slot.thermostat->fixed_group != -1 || slot.thermostat->move_group != -1)
      PRINT_INPUT_ERROR("Replicas in one batch must share the thermostat settings.");
    host_target_temperature_.data()[i] = slot.thermostat->temperature;
  }

  // Order the batch after any pending work on the slot streams.
  for (Replica_Slot* slot : slots) {
    slot->complete.record(slot->stream);
    stream_.wait(slot->complete);
  }
  gather(slots, total);
  target_temperature_.copy_from_host_async(host_target_temperature_.data(), replicas, stream_.get());

  Batch_Box batch_box;
  batch_box.box = box_;
  batch_box.atoms_per_replica = N;
  const double volume = box_.get_volume();
  const int threads = 128;
  const int blocks = (total - 1) / threads + 1;

  int done = 0;
  for (int chunk = 0; done < steps; ++chunk) {
    const int b = chunk % 2;
    const int count = std::min(chunk_steps_, steps - done);
    ++statistics_.noise_chunks;
    if (chunk >= 2) {
      chunk_done_[b].synchronize(); // host_noise_[b] is reused
      ++statistics_.noise_buffer_reuses;
    }
    if (chunk > 0 && count < chunk_steps_)
      ++statistics_.partial_tail_chunks;
    double* host_noise = host_noise_[b].data();
    for (int s = 0; s < count; ++s) {
      for (int i = 0; i < replicas; ++i) {
        double* value = host_noise + (static_cast<size_t>(s) * replicas + i) * 2;
        slots[i]->thermostat->draw_svr_noise(ndeg, value[0], value[1]);
      }
    }
    noise_[b].copy_from_host_async(host_noise, static_cast<size_t>(count) * replicas * 2, stream_.get());

    for (int s = 0; s < count; ++s) {
      gpu_velocity_verlet_batched<<<blocks, threads, 0, stream_.get()>>>(
        true, total, time_step, mass_.data(),
        position_.data(), position_.data() + total, position_.data() + total * 2,
        velocity_.data(), velocity_.data() + total, velocity_.data() + total * 2,
        force_.data(), force_.data() + total, force_.data() + total * 2);
      GPU_CHECK_KERNEL
      gpu_prepare_force_batched<<<blocks, threads, 0, stream_.get()>>>(
        total, box_,
        position_.data(), position_.data() + total, position_.data() + total * 2,
        force_.data(), force_.data() + total, force_.data() + total * 2,
        potential_.data(), virial_.data());
      GPU_CHECK_KERNEL
      nep_->compute_batched(
        batch_box, total, type_.data(), position_.data(), potential_.data(),
        force_.data(), virial_.data(), stream_.get());
      gpu_velocity_verlet_batched<<<blocks, threads, 0, stream_.get()>>>(
        false, total, time_step, mass_.data(),
        position_.data(), position_.data() + total, position_.data() + total * 2,
        velocity_.data(), velocity_.data() + total, velocity_.data() + total * 2,
        force_.data(), force_.data() + total, force_.data() + total * 2);
      GPU_CHECK_KERNEL
      gpu_find_thermo_batched<<<dim3(8, replicas), 1024, 0, stream_.get()>>>(
        N, ndeg, 0.0, volume, mass_.data(), potential_.data(),
        velocity_.data(), velocity_.data() + total, velocity_.data() + total * 2,
        virial_.data(), virial_.data() + total, virial_.data() + total * 2,
        virial_.data() + total * 3, virial_.data() + total * 4, virial_.data() + total * 5,
        thermo_.data());
      GPU_CHECK_KERNEL
      gpu_resamplekin_and_scale_batched<<<blocks, threads, 0, stream_.get()>>>(
        total, N, ndeg, coupling, target_temperature_.data(),
        noise_[b].data() + static_cast<size_t>(s) * replicas * 2, thermo_.data(),
        velocity_.data(), velocity_.data() + total, velocity_.data() + total * 2);
      GPU_CHECK_KERNEL
    }
    chunk_done_[b].record(stream_);
    done += count;
  }

  scatter(slots, total);
  thermo_.copy_to_host_async(host_thermo_.data(), static_cast<size_t>(replicas) * 8, stream_.get());
  stream_.synchronize();
  for (int i = 0; i < replicas; ++i) {
    Replica_Slot& slot = *slots[i];
    slot.host_thermo.data()[0] = host_thermo_.data()[i * 8];
    slot.host_thermo.data()[1] = host_thermo_.data()[i * 8 + 1];
    slot.potential_energy = host_thermo_.data()[i * 8 + 1];
  }
}

} // namespace replica
