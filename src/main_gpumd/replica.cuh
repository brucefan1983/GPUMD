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
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

constexpr double DEFAULT_REMD_TARGET_ACCEPTANCE = 0.2;

struct MultiReplicaConfig
{
  enum Mode { none, remd };
  enum TemperatureStyle { geometric, linear };

  Mode mode = none;
  TemperatureStyle temperature_style = geometric;
  int replicas = 0;
  int exchange_interval = 0;
  int equilibration_steps = 0;
  int seed = 0;
  int replicas_per_gpu = 0;
  std::vector<int> devices;
  std::vector<double> temperatures;
  std::vector<double> betas;
  bool has_temperature_range = false;
  bool generated_temperatures = false;
  bool verbose_output = false;
  bool resume = false;
  bool dump_xyz = false;
  int dump_xyz_interval = 0;
  int dump_xyz_has_velocity = 0;
  std::string resume_prefix;
  std::string dump_xyz_filename;
  double temperature_min = 0.0;
  double temperature_max = 0.0;

  void parse(const char** param, int num_param);
  void parse_dump_xyz(const char** param, int num_param);
  bool enabled() const { return mode != none; }
};

struct ReplicaContext
{
  ~ReplicaContext();

  int slot_id = 0;
  int device_id = 0;
  double target_temperature = 0.0;
  double potential_energy = 0.0;
  double timing_compute1 = 0.0;
  double timing_force = 0.0;
  double timing_compute2 = 0.0;
  double timing_thermo_copy = 0.0;

  Atom atom;
  Box box;
  std::vector<Group> group;
  GPU_Vector<double> thermo;
  Force force;
  Integrate integrate;
};

struct LogicalReplicaState
{
  int replica_id = 0;
  int temperature_label = 0;
  double target_temperature = 0.0;
  double potential_energy = 0.0;
  Box box;
  std::vector<double> position_per_atom;
  std::vector<double> velocity_per_atom;
  std::vector<double> ensemble_state;
  bool has_ensemble_state = false;
};

struct WorkerSlot
{
  std::thread thread;
  std::mutex mtx;
  std::condition_variable cv;
  int logical_id = -1;
  int start_step = 0;
  int steps = 0;
  int total_steps = -1;
  bool ready = false;
  bool finished = false;
  bool stop = false;
};

struct MultiReplicaTiming
{
  double initialize = 0.0;
  double propagate = 0.0;
  double exchange = 0.0;
  double output = 0.0;
  double restart = 0.0;
  double total = 0.0;
};

class MultiReplicaDriver
{
public:
  MultiReplicaDriver(
    MultiReplicaConfig& multi_replica,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    const std::vector<std::vector<std::string>>& potential_commands,
    const std::vector<std::string>& ensemble_command,
    double time_step,
    int number_of_steps);

  void run();

private:
  MultiReplicaConfig& config_;
  Atom& source_atom_;
  Box& source_box_;
  std::vector<Group>& source_group_;
  const std::vector<std::vector<std::string>>& potential_commands_;
  const std::vector<std::string>& ensemble_command_;
  double time_step_;
  int number_of_steps_;

  std::vector<LogicalReplicaState> logical_replicas_;
  std::vector<std::unique_ptr<ReplicaContext>> gpu_slots_;
  std::vector<std::unique_ptr<WorkerSlot>> workers_;
  std::vector<std::unique_ptr<WorkerSlot>> resident_workers_;
  std::vector<int> replica_to_temperature_;
  std::vector<int> temperature_to_replica_;
  std::vector<int> exchange_attempts_;
  std::vector<int> exchange_accepts_;
  std::vector<double> exchange_probability_sum_;
  std::vector<FILE*> remd_dump_xyz_files_;
  std::vector<double> remd_dump_position_;
  std::vector<double> remd_dump_velocity_;
  int remd_exchange_count_ = 0;

  MultiReplicaTiming timing_;

  void initialize_replicas();
  void initialize_logical_replicas();
  void read_remd_restart();
  void read_remd_restart_metadata(const std::string& filename);
  void read_remd_restart_xyz(
    const std::string& filename, int replica_id, LogicalReplicaState& logical_replica);
  void validate_remd_restart_metadata() const;
  void initialize_gpu_slots();
  void initialize_one_gpu_slot(ReplicaContext& replica);
  void validate_ensemble_support(ReplicaContext& replica);
  void run_remd();
  void run_remd_batched();
  void run_remd_resident();
  void equilibrate_remd_batched();
  void equilibrate_remd_resident();
  void open_remd_logs(FILE*& exchange_log, FILE*& status_log, FILE*& thermo_log) const;
  void close_remd_logs(FILE* exchange_log, FILE* status_log, FILE* thermo_log) const;
  void write_remd_thermo(int step, FILE* thermo_log) const;
  void open_remd_xyz_dump();
  void close_remd_xyz_dump();
  void write_remd_xyz_dump(int step, bool resident_mode);
  void print_remd_progress(int step) const;
  void print_remd_performance(double time_used) const;
  void write_remd_outputs(FILE* exchange_log);
  bool can_use_resident_remd() const;
  void initialize_resident_slots();
  void advance_resident_slots(int start_step, int steps, int total_steps = -1);
  void attempt_exchanges_resident(int exchange_count, int step, FILE* exchange_log);
  void scale_slot_velocity(ReplicaContext& slot, double factor);
  void update_resident_replica_temperature(int replica_id, int new_label);
  void resident_worker_loop(int slot_id);
  void start_resident_workers();
  void stop_resident_workers();
  void worker_loop(int slot_id);
  void start_workers();
  void stop_workers();
  void advance_replica(ReplicaContext& replica, int start_step, int steps, int total_steps = -1);
  void advance_logical_replicas(
    const std::vector<int>& logical_ids, int start_step, int steps, int total_steps = -1);
  void compute_initial_force(ReplicaContext& replica);
  void set_slot_temperature(ReplicaContext& replica, int temperature_label);
  void export_ensemble_state(ReplicaContext& slot, LogicalReplicaState& logical_replica);
  void import_ensemble_state(LogicalReplicaState& logical_replica, ReplicaContext& slot);
  double get_temperature_from_label(int temperature_label) const;
  void load_logical_replica_to_slot(LogicalReplicaState& logical_replica, ReplicaContext& slot);
  void save_slot_to_logical_replica(ReplicaContext& slot, LogicalReplicaState& logical_replica);
  void randomize_logical_replica_velocity(LogicalReplicaState& logical_replica, int seed);
  void scale_logical_replica_velocity(LogicalReplicaState& logical_replica, double factor);
  void update_logical_replica_temperature(LogicalReplicaState& logical_replica, int temperature_label);
  void attempt_exchanges(int exchange_count, int step, FILE* exchange_log);
  void write_exchange_statistics(FILE* exchange_log);
  void write_temperature_ladder_file();
  void write_ladder_report();
  void write_temperature_status(int step, FILE* status_log);
  void write_restart_files(const char* mode_name);
  void write_remd_restart_metadata(const char* mode_name);
  void print_timing_summary(const char* mode_name) const;
};

namespace replica {

void reset();
bool enabled();
void parse(const char** param, int num_param);
void remember_potential_command(const std::vector<std::string>& tokens);
void remember_ensemble_command(const std::vector<std::string>& tokens);
void remember_dump_xyz_command(const std::vector<std::string>& tokens);
bool run_if_enabled(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  double time_step,
  int number_of_steps,
  double max_distance_per_step);

} // namespace replica
