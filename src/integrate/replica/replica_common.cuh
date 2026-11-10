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

#pragma once

#include "ensemble_bdp_replica.cuh"
#include "force/replica/replica_force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_pinned_vector.cuh"
#include "utilities/gpu_stream.cuh"
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <istream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

class RunInput;

namespace replica
{

class Replica_Batch;

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
  std::unique_ptr<Ensemble_BDP_Replica> thermostat;
  GPU_Pinned_Vector<double> host_thermo;
};

bool values_match(double first, double second);
bool file_has_content(const std::string& filename);
std::string trim_left(const std::string& value);
unsigned long long hash_file(const std::string& filename);
bool extract_quoted_value(
  const std::string& line, const std::string& key, std::string& value);
bool extract_token_value(
  const std::string& line, const std::string& key, std::string& value);
void validate_xyz_box(
  const std::string& comment, const Box& reference, const char* mode);

// Replica restart xyz files store each velocity twice: vel in A/fs for other
// tools and vel_natural in GPUMD's internal unit. Converting A/fs back is not
// exact in the last bit, which would make a resumed run leave the trajectory
// of the uninterrupted one. Files without vel_natural are still accepted.
extern const char RESTART_XYZ_PROPERTIES[];
bool parse_restart_xyz_properties(const std::string& properties, bool& natural_velocity);
void write_restart_xyz_velocity(FILE* file, const std::vector<double>& velocity, int atom);
bool read_restart_xyz_velocity(
  std::istream& input, bool natural_velocity, std::vector<double>& velocity, int atom);

template <typename T>
bool parse_scalar(const std::string& text, T& value)
{
  std::istringstream input(text);
  std::string trailing;
  return (input >> value) && !(input >> trailing);
}

// Input handling shared by the REMD and PRD drivers. `mode` ("REMD" or "PRD") starts the error
// messages.

// replicas N [replicas_per_gpu] of multi_replica, with command[i] = "replicas".
void parse_replicas_option(
  const std::vector<std::string>& command,
  size_t& i,
  int minimum_replicas,
  const char* mode,
  int& replicas,
  int& replicas_per_gpu);
// dump_xyz -1 group interval filename [velocity], the velocity only if allowed.
void parse_dump_xyz(
  const std::vector<std::string>& dump_command,
  const char* mode,
  bool allow_velocity,
  int& dump_interval,
  std::string& dump_prefix,
  bool& dump_velocity);
unsigned int draw_internal_seed();
// One NEP or qNEP potential; kspace_method becomes "none" for NEP. Returns the file hash.
unsigned long long validate_potential(
  const std::vector<std::vector<std::string>>& potential_commands,
  const Box& box,
  const char* mode,
  std::string& kspace_method);
// ensemble nvt_bdp T1 T2 tau_T
void parse_nvt_bdp(
  const std::vector<std::string>& ensemble_command,
  const char* mode,
  double& temperature_1,
  double& temperature_2,
  double& temperature_coupling);
// The default is ceil(replicas / GPUs).
int resolve_replicas_per_gpu(int replicas, int replicas_per_gpu, const char* mode);
void initialize_thermostat_rng(
  Replica_Slot& slot,
  bool resume,
  const std::string& restart_state,
  unsigned int internal_seed,
  const char* mode);

// Output files: append when resuming, otherwise start a new file. `fresh` tells whether the file
// is new, in which case the marker line has been written and the caller adds its header.
void validate_append_marker(const std::string& filename, const char* marker, const char* mode);
FILE* open_output_file(const std::string& filename, bool resume, const char* marker, bool& fresh);

// pbc="..." Lattice="..." of an extended xyz comment line, each followed by a space.
void write_xyz_box(FILE* file, const Box& box, int precision);
// A replica restart xyz with the positions and velocities of the slot; `fields` follow
// Properties on the comment line.
void write_restart_xyz(
  const std::string& filename,
  Replica_Slot& slot,
  const Atom& source_atom,
  const std::string& fields);
// Checks the atom count and the box of a restart xyz and returns its comment line.
std::string read_restart_xyz_header(
  std::ifstream& input,
  const std::string& filename,
  const Atom& source_atom,
  const Box& source_box,
  const char* what);
// The atoms of a replica restart xyz, into atom.cpu_position_per_atom and cpu_velocity_per_atom.
void read_restart_xyz_atoms(
  std::istream& input,
  const std::string& comment,
  const Atom& source_atom,
  Atom& atom,
  const char* mode);

// Restart metadata: keys in the order in which the drivers write them.
void read_restart_metadata_header(
  std::istream& input, const char* mode, const char* mode_key, int format_version);
void read_restart_key(std::istream& input, const char* mode, const char* expected_key);
template <typename T>
void read_restart_value(std::istream& input, const char* mode, const char* expected_key, T& value)
{
  read_restart_key(input, mode, expected_key);
  if (!(input >> value)) {
    const std::string message =
      std::string(mode) + " restart value of " + expected_key + " is invalid.";
    PRINT_INPUT_ERROR(message.c_str());
  }
}
// The rest of the line after the key, without surrounding white space.
std::string read_restart_line(std::istream& input, const char* mode, const char* expected_key);
// bdp_rng lines of replicas 0 to replicas - 1, then the end of the file.
void read_restart_bdp_states(
  std::istream& input, const char* mode, int replicas, std::vector<std::string>& states);

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
  Replica_Force force;
  std::unique_ptr<Replica_Batch> batch; // set when batched integration is enabled
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
    const RunInput& run_input,
    Atom& source_atom,
    Box& source_box,
    std::vector<Group>& source_group,
    double time_step);
  ~Replica_Runtime();

  Replica_Runtime(const Replica_Runtime&) = delete;
  Replica_Runtime& operator=(const Replica_Runtime&) = delete;

  // Batched integration (one concatenated system per GPU) is chosen in initialize: on when a GPU
  // holds at least two replicas and at most 20000 atoms in total.
  void initialize(int replicas, int replicas_per_gpu, const Slot_Initializer& initialize_slot);
  void start_workers();
  void stop_workers();
  void advance_all(int steps);
  void advance_selected(const std::vector<int>& replica_ids, int steps);
  void compute_forces_all();
  void compute_forces_selected(const std::vector<int>& replica_ids);
  Replica_Force& force(Replica_Slot& slot);

  std::vector<Replica_Slot*>& slots() { return replica_slots_; }
  const std::vector<Replica_Slot*>& slots() const { return replica_slots_; }

private:
  const std::vector<std::vector<std::string>>& potential_commands_;
  const RunInput& run_input_;
  Atom& source_atom_;
  Box& source_box_;
  std::vector<Group>& source_group_;
  double time_step_;
  bool batch_active_ = false;
  std::vector<Replica_Slot*> replica_slots_;
  std::vector<std::unique_ptr<Device_Context>> devices_;

  void clone_source_state(Replica_Slot& slot) const;
  void initialize_device_force(Device_Context& device);
  void initialize_batches();
  void report_batches() const;
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
