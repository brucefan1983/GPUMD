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

#include "force/force.cuh"
#include "ensemble_bdp.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include "utilities/gpu_stream.cuh"
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace replica
{

struct Replica_Slot {
  Replica_Slot(int replica_id, int device_id);
  ~Replica_Slot();

  Replica_Slot(const Replica_Slot&) = delete;
  Replica_Slot& operator=(const Replica_Slot&) = delete;

  int replica_id;
  int device_id;
  int temperature_label = 0;
  double target_temperature = 0.0;
  double potential_energy = 0.0;
  Box box;
  GPU_Stream stream;
  GPU_Event complete;
  Atom atom;
  std::vector<Group> group;
  GPU_Vector<double> thermo;
  std::unique_ptr<Ensemble_BDP> thermostat;
  GPU_Pinned_Vector<double> host_thermo;
};

bool values_match(double first, double second);
bool file_has_content(const std::string& filename);
std::string trim_left(const std::string& value);
unsigned long long hash_file(const std::string& filename);
std::string read_potential_name(const std::string& filename);
bool is_supported_energy_nep_name(const std::string& name);
bool extract_quoted_value(
  const std::string& line, const std::string& key, std::string& value);
bool extract_token_value(
  const std::string& line, const std::string& key, std::string& value);
void validate_xyz_box(
  const std::string& comment, const Box& reference, const char* mode);

template <typename T>
bool parse_scalar(const std::string& text, T& value)
{
  std::istringstream input(text);
  std::string trailing;
  return (input >> value) && !(input >> trailing);
}

class Replica_State
{
public:
  Replica_State(int device_id, int number_of_atoms, bool store_velocity);
  ~Replica_State();

  Replica_State(const Replica_State&) = delete;
  Replica_State& operator=(const Replica_State&) = delete;

  void save(Replica_Slot& slot);
  void restore(Replica_Slot& slot);
  void copy_position_to_host(
    GPU_Pinned_Vector<double>& host_position, Replica_Slot& slot);
  void copy_position_from_host(
    const GPU_Pinned_Vector<double>& host_position, Replica_Slot& slot);

  const GPU_Vector<double>& position() const { return position_; }
  GPU_Vector<double>& position() { return position_; }
  bool stores_velocity() const { return store_velocity_; }

private:
  int device_id_;
  bool store_velocity_;
  GPU_Vector<double> position_;
  GPU_Vector<double> velocity_;
};

struct Device_Context {
  explicit Device_Context(int device_id);
  ~Device_Context();

  Device_Context(const Device_Context&) = delete;
  Device_Context& operator=(const Device_Context&) = delete;

  int device_id;
  Force force;
  std::vector<std::unique_ptr<Replica_Slot>> slots;
  std::thread worker;
  std::mutex mutex;
  std::condition_variable condition;
  std::vector<unsigned char> active_slots;
  bool ready = false;
  bool finished = false;
  bool stop = false;
  int steps = 0;
};

class Replica_Runtime
{
public:
  using Slot_Initializer = std::function<void(Replica_Slot&)>;

  Replica_Runtime(
    const std::vector<std::vector<std::string>>& potential_commands,
    Atom& source_atom,
    Box& source_box,
    std::vector<Group>& source_group,
    double time_step);
  ~Replica_Runtime();

  Replica_Runtime(const Replica_Runtime&) = delete;
  Replica_Runtime& operator=(const Replica_Runtime&) = delete;

  void initialize(int replicas, int replicas_per_gpu, const Slot_Initializer& initialize_slot);
  void start_workers();
  void stop_workers();
  void advance_all(int steps);
  void advance_selected(const std::vector<int>& replica_ids, int steps);
  void compute_force(Replica_Slot& slot);
  void compute_forces_all();
  void compute_forces_selected(const std::vector<int>& replica_ids);
  Force& force(Replica_Slot& slot);

  std::vector<Replica_Slot*>& slots() { return replica_slots_; }
  const std::vector<Replica_Slot*>& slots() const { return replica_slots_; }

private:
  const std::vector<std::vector<std::string>>& potential_commands_;
  Atom& source_atom_;
  Box& source_box_;
  std::vector<Group>& source_group_;
  double time_step_;
  std::vector<Replica_Slot*> replica_slots_;
  std::vector<std::unique_ptr<Device_Context>> devices_;

  void clone_source_state(Replica_Slot& slot) const;
  void initialize_device_force(Device_Context& device);
  void compute_force(Device_Context& device, Replica_Slot& slot);
  void worker_loop(Device_Context& device);
  void advance_device(
    Device_Context& device,
    const std::vector<unsigned char>& active_slots,
    int steps);
  void advance(const std::vector<unsigned char>& active_replicas, int steps);
  void compute_forces(const std::vector<unsigned char>& active_replicas);
};

} // namespace replica
