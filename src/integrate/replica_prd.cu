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

/*----------------------------------------------------------------------------80
Parallel replica dynamics for infrequent-event systems. Basin identities are
defined by FIRE quenches and an atom-displacement event criterion.
------------------------------------------------------------------------------*/

#include "replica.cuh"
#include "replica_common.cuh"
#include "main_gpumd/velocity.cuh"
#include "minimize/minimizer_fire.cuh"
#include "model/read_xyz.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
using replica::Replica_Runtime;
using replica::Replica_Slot;
using replica::Replica_State;
using replica::extract_token_value;
using replica::file_has_content;
using replica::hash_file;
using replica::is_supported_energy_nep_name;
using replica::parse_scalar;
using replica::read_potential_name;
using replica::trim_left;
using replica::validate_xyz_box;
using replica::values_match;

constexpr const char* event_output_marker =
  "# GPUMD PRD event output format 1";
constexpr int restart_format_version = 3;

void validate_append_event_file(const std::string& filename)
{
  if (!file_has_content(filename))
    return;
  std::ifstream input(filename);
  std::string first_line;
  if (!std::getline(input, first_line) ||
      first_line != event_output_marker) {
    const std::string message =
      "PRD cannot append to " + filename +
      " because its output format is incompatible.";
    PRINT_INPUT_ERROR(message.c_str());
  }
}

template <typename T>
void read_restart_value(
  std::ifstream& input, const char* expected_key, T& value)
{
  std::string key;
  if (!(input >> key >> value) || key != expected_key) {
    const std::string message =
      "Invalid or missing PRD restart key " +
      std::string(expected_key) + ".";
    PRINT_INPUT_ERROR(message.c_str());
  }
}

std::string read_restart_line(
  std::ifstream& input, const char* expected_key)
{
  std::string key;
  std::string value;
  if (!(input >> key) || key != expected_key) {
    const std::string message =
      "Invalid or missing PRD restart key " +
      std::string(expected_key) + ".";
    PRINT_INPUT_ERROR(message.c_str());
  }
  std::getline(input, value);
  value = trim_left(value);
  if (value.empty()) {
    const std::string message =
      "Empty PRD restart value for " + std::string(expected_key) + ".";
    PRINT_INPUT_ERROR(message.c_str());
  }
  return value;
}

void validate_append_trajectory_file(
  const std::string& filename, const int number_of_atoms)
{
  if (!file_has_content(filename))
    return;
  std::ifstream input(filename);
  int stored_atoms = 0;
  std::string comment;
  std::string value;
  if (!(input >> stored_atoms) || stored_atoms != number_of_atoms) {
    PRINT_INPUT_ERROR(
      "PRD cannot append to an incompatible event trajectory.");
  }
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  if (!std::getline(input, comment) ||
      !extract_token_value(comment, "replica_mode", value) ||
      value != "prd_event")
    PRINT_INPUT_ERROR(
      "PRD cannot append to an incompatible event trajectory.");
}

int make_positive_seed(const unsigned int seed)
{
  const unsigned int limit =
    static_cast<unsigned int>(std::numeric_limits<int>::max() - 1);
  return static_cast<int>(seed % limit) + 1;
}

struct PRD_Config {
  int replicas = 0;
  int replicas_per_gpu = 0;
  int event_interval = 0;
  int dephase_iterations = -1;
  int dephase_steps = 0;
  int correlation_steps = -1;
  int quench_steps = 1000;
  int maximum_dephase_retries = 1000;
  double event_distance = 0.0;
  double quench_force_tolerance = 1.0e-4;
  bool resume = false;
  bool dump_xyz = false;
  int dump_interval = 0;
  std::string resume_prefix;
  std::string dump_prefix;
  unsigned int internal_seed = 1;

  void parse(
    const std::vector<std::string>& command,
    const std::vector<std::string>& dump_command);
};

void PRD_Config::parse(
  const std::vector<std::string>& command,
  const std::vector<std::string>& dump_command)
{
  if (command.size() < 3 || command[0] != "multi_replica" ||
      command[1] != "prd")
    PRINT_INPUT_ERROR("Invalid multi_replica prd command.");
  for (size_t i = 2; i < command.size();) {
    if (command[i] == "replicas") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &replicas) || replicas <= 0)
        PRINT_INPUT_ERROR("PRD replicas should be a positive integer.");
      ++i;
      if (i < command.size()) {
        int value = 0;
        if (is_valid_int(command[i].c_str(), &value)) {
          if (value <= 0)
            PRINT_INPUT_ERROR(
              "PRD replicas_per_gpu should be positive.");
          replicas_per_gpu = value;
          ++i;
        }
      }
    } else if (
      command[i] == "event" || command[i] == "event_interval") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &event_interval) ||
          event_interval <= 0)
        PRINT_INPUT_ERROR(
          "PRD event interval should be a positive integer.");
      ++i;
    } else if (command[i] == "dephase") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &dephase_iterations) ||
          dephase_iterations < 0 || ++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &dephase_steps) ||
          dephase_steps <= 0)
        PRINT_INPUT_ERROR(
          "PRD dephase requires non-negative iterations and positive steps.");
      ++i;
    } else if (
      command[i] == "correlate" || command[i] == "correlation") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &correlation_steps) ||
          correlation_steps < 0)
        PRINT_INPUT_ERROR(
          "PRD correlation steps should be a non-negative integer.");
      ++i;
    } else if (
      command[i] == "distance" || command[i] == "event_distance") {
      if (++i >= command.size() ||
          !is_valid_real(command[i].c_str(), &event_distance) ||
          event_distance <= 0.0)
        PRINT_INPUT_ERROR(
          "PRD event distance should be a positive real number.");
      ++i;
    } else if (command[i] == "quench") {
      if (++i >= command.size() ||
          !is_valid_real(command[i].c_str(), &quench_force_tolerance) ||
          quench_force_tolerance <= 0.0 || ++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &quench_steps) ||
          quench_steps <= 0)
        PRINT_INPUT_ERROR(
          "PRD quench requires a positive force tolerance and step count.");
      ++i;
    } else if (command[i] == "max_dephase_retries") {
      if (++i >= command.size() ||
          !is_valid_int(
            command[i].c_str(), &maximum_dephase_retries) ||
          maximum_dephase_retries <= 0)
        PRINT_INPUT_ERROR(
          "PRD max_dephase_retries should be a positive integer.");
      ++i;
    } else if (command[i] == "resume") {
      if (resume)
        PRINT_INPUT_ERROR("Specify PRD resume only once.");
      if (++i >= command.size())
        PRINT_INPUT_ERROR("PRD resume requires a restart prefix.");
      resume = true;
      resume_prefix = command[i++];
    } else if (
      command[i] == "verbose" || command[i] == "verbose_output") {
      ++i;
    } else {
      PRINT_INPUT_ERROR("Unknown multi_replica prd option.");
    }
  }

  if (
    replicas <= 0 || event_interval <= 0 || dephase_iterations < 0 ||
    correlation_steps < 0 || event_distance <= 0.0)
    PRINT_INPUT_ERROR(
      "PRD requires replicas, event, dephase, correlate, and distance.");

  if (!dump_command.empty()) {
    if (dump_command.size() < 5 || dump_command[0] != "dump_xyz")
      PRINT_INPUT_ERROR("Invalid PRD dump_xyz command.");
    int grouping_method = 0;
    int group_id = 0;
    if (!is_valid_int(dump_command[1].c_str(), &grouping_method) ||
        grouping_method != -1 ||
        !is_valid_int(dump_command[2].c_str(), &group_id))
      PRINT_INPUT_ERROR("PRD dump_xyz requires grouping_method -1.");
    if (!is_valid_int(
          dump_command[3].c_str(), &dump_interval) ||
        dump_interval <= 0)
      PRINT_INPUT_ERROR("PRD dump_xyz interval should be positive.");
    dump_prefix = dump_command[4];
    if (dump_prefix.empty() || dump_prefix.back() == '*')
      PRINT_INPUT_ERROR(
        "PRD dump_xyz does not support an empty or '*' filename.");
    if (dump_command.size() != 5)
      PRINT_INPUT_ERROR(
        "PRD dump_xyz supports quenched positions only.");
    dump_xyz = true;
  }

  internal_seed = static_cast<unsigned int>(rand());
  if (internal_seed == 0)
    internal_seed = 1;
}

static __device__ void apply_prd_mic(
  const Box& box, double& dx, double& dy, double& dz)
{
  if (!box.is_orthogonal) {
    apply_mic(box, dx, dy, dz);
    return;
  }
  if (box.pbc_x == 1)
    dx -= nearbyint(dx / box.cpu_h[0]) * box.cpu_h[0];
  if (box.pbc_y == 1)
    dy -= nearbyint(dy / box.cpu_h[4]) * box.cpu_h[4];
  if (box.pbc_z == 1)
    dz -= nearbyint(dz / box.cpu_h[8]) * box.cpu_h[8];
}

static __global__ void gpu_maximum_displacement_squared(
  const int number_of_atoms,
  const Box box,
  const double* position,
  const double* reference_position,
  double* maximum_squared)
{
  const int thread = threadIdx.x;
  double local_maximum = 0.0;
  for (int atom = thread; atom < number_of_atoms; atom += blockDim.x) {
    double dx = position[atom] - reference_position[atom];
    double dy =
      position[atom + number_of_atoms] -
      reference_position[atom + number_of_atoms];
    double dz =
      position[atom + 2 * number_of_atoms] -
      reference_position[atom + 2 * number_of_atoms];
    apply_prd_mic(box, dx, dy, dz);
    local_maximum = fmax(local_maximum, dx * dx + dy * dy + dz * dz);
  }

  __shared__ double maximum[256];
  maximum[thread] = local_maximum;
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (thread < offset)
      maximum[thread] = fmax(maximum[thread], maximum[thread + offset]);
    __syncthreads();
  }
  if (thread == 0)
    maximum_squared[0] = maximum[0];
}

struct PRD_Slot_Data {
  PRD_Slot_Data(
    Replica_Slot& slot,
    const int quench_steps,
    const double force_tolerance)
    : device_id(slot.device_id)
    , hot(slot.device_id, slot.atom.number_of_atoms, true)
    , dephase_start(slot.device_id, slot.atom.number_of_atoms, true)
    , block_start(slot.device_id, slot.atom.number_of_atoms, true)
    , basin(slot.device_id, slot.atom.number_of_atoms, false)
    , minimizer(slot.atom.number_of_atoms, quench_steps, force_tolerance)
    , maximum_displacement_squared(1)
    , host_maximum_displacement_squared(1, 0.0)
  {
  }

  ~PRD_Slot_Data() { CHECK(gpuSetDevice(device_id)); }

  int device_id;
  Replica_State hot;
  Replica_State dephase_start;
  Replica_State block_start;
  Replica_State basin;
  Minimizer_FIRE minimizer;
  GPU_Vector<double> maximum_displacement_squared;
  GPU_Pinned_Vector<double> host_maximum_displacement_squared;
  BDP_RNG_State block_rng_state;
  Minimization_Result quench_result;
  double maximum_displacement = 0.0;
};

struct Search_Result {
  bool event = false;
  int winner = -1;
  int steps = 0;
  int coincident = 0;
  double maximum_displacement = 0.0;
  Minimization_Result quench;
};

enum class PRD_Phase { dephase, parallel, correlation };

const char* phase_name(const PRD_Phase phase)
{
  if (phase == PRD_Phase::dephase)
    return "dephase";
  if (phase == PRD_Phase::parallel)
    return "parallel";
  return "correlation";
}

class PRD_Driver
{
public:
  PRD_Driver(
    const std::vector<std::vector<std::string>>& potential_commands,
    const std::vector<std::string>& ensemble_command,
    const std::vector<std::string>& multi_replica_command,
    const std::vector<std::string>& dump_xyz_command,
    Atom& source_atom,
    Box& source_box,
    std::vector<Group>& source_group,
    double time_step,
    int number_of_steps);
  ~PRD_Driver();

  void run();

private:
  PRD_Config config_;
  const std::vector<std::vector<std::string>>& potential_commands_;
  const std::vector<std::string>& ensemble_command_;
  Atom& source_atom_;
  Box& source_box_;
  std::vector<Group>& source_group_;
  double time_step_;
  int number_of_steps_;
  double temperature_ = 0.0;
  double temperature_coupling_ = 0.0;
  unsigned long long potential_hash_ = 0;
  long long completed_dynamics_steps_ = 0;
  int dynamics_steps_ = 0;
  long long clock_ = 0;
  int event_number_ = 0;
  int dephase_rejections_ = 0;
  PRD_Phase phase_ = PRD_Phase::dephase;
  int correlation_winner_ = -1;
  int correlation_quiet_steps_ = 0;
  FILE* event_file_ = nullptr;
  FILE* trajectory_file_ = nullptr;
  bool trajectory_had_content_ = false;
  Replica_Runtime runtime_;
  std::vector<std::unique_ptr<PRD_Slot_Data>> slot_data_;
  std::vector<std::string> restart_bdp_states_;
  GPU_Pinned_Vector<double> basin_position_;
  GPU_Pinned_Vector<double> hot_position_;
  std::mt19937 selection_rng_;
  std::mt19937 dephase_rng_;

  void validate_input();
  void initialize_slot(Replica_Slot& slot);
  void initialize_runtime();
  void initialize_basin();
  void open_output();
  void close_output();
  long long global_dynamics_step() const;
  std::vector<int> all_replicas() const;
  Ensemble_BDP& bdp(int replica_id);
  void save_hot(const std::vector<int>& replica_ids);
  void restore_hot(const std::vector<int>& replica_ids);
  void save_block_start(const std::vector<int>& replica_ids);
  void restore_block_start(const std::vector<int>& replica_ids);
  void quench(const std::vector<int>& replica_ids);
  std::vector<int> detect_events(
    const std::vector<int>& replica_ids);
  void broadcast_basin(int replica_id);
  void broadcast_hot_position(int replica_id);
  void randomize_velocities(const std::vector<int>& replica_ids);
  void dephase();
  Search_Result search_block(
    const std::vector<int>& replica_ids, int maximum_steps);
  void accept_event(const Search_Result& result, bool correlated);
  void write_event(const Search_Result& result, bool correlated);
  void write_event_trajectory(int replica_id, bool correlated);
  void finish_correlation();
  void write_restart();
  void write_restart_metadata() const;
  void write_restart_xyz(Replica_Slot& slot) const;
  void write_restart_basin() const;
  void read_restart_metadata();
  void read_restart_xyz(Replica_Slot& slot);
  void read_restart_basin();
};

PRD_Driver::PRD_Driver(
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  const std::vector<std::string>& multi_replica_command,
  const std::vector<std::string>& dump_xyz_command,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  const double time_step,
  const int number_of_steps)
  : potential_commands_(potential_commands)
  , ensemble_command_(ensemble_command)
  , source_atom_(source_atom)
  , source_box_(source_box)
  , source_group_(source_group)
  , time_step_(time_step)
  , number_of_steps_(number_of_steps)
  , runtime_(
      potential_commands,
      source_atom,
      source_box,
      source_group,
      time_step)
{
  config_.parse(multi_replica_command, dump_xyz_command);
}

PRD_Driver::~PRD_Driver()
{
  runtime_.stop_workers();
  close_output();
  slot_data_.clear();
}

std::vector<int> PRD_Driver::all_replicas() const
{
  std::vector<int> replica_ids(config_.replicas);
  for (int replica_id = 0; replica_id < config_.replicas; ++replica_id)
    replica_ids[replica_id] = replica_id;
  return replica_ids;
}

Ensemble_BDP& PRD_Driver::bdp(const int replica_id)
{
  Replica_Slot& slot = *runtime_.slots()[replica_id];
  return *slot.thermostat;
}

void PRD_Driver::validate_input()
{
  if (number_of_steps_ <= 0)
    PRINT_INPUT_ERROR("PRD run steps should be positive.");
  if (source_atom_.number_of_atoms <= 0)
    PRINT_INPUT_ERROR("PRD requires a non-empty model.");
  double minimum_periodic_thickness =
    std::numeric_limits<double>::infinity();
  if (source_box_.pbc_x)
    minimum_periodic_thickness =
      std::min(
        minimum_periodic_thickness,
        source_box_.get_volume() / source_box_.get_area(0));
  if (source_box_.pbc_y)
    minimum_periodic_thickness =
      std::min(
        minimum_periodic_thickness,
        source_box_.get_volume() / source_box_.get_area(1));
  if (source_box_.pbc_z)
    minimum_periodic_thickness =
      std::min(
        minimum_periodic_thickness,
        source_box_.get_volume() / source_box_.get_area(2));
  if (
    std::isfinite(minimum_periodic_thickness) &&
    config_.event_distance >= 0.5 * minimum_periodic_thickness)
    PRINT_INPUT_ERROR(
      "PRD event distance should be less than half the shortest "
      "periodic box thickness.");
  if (potential_commands_.size() != 1 ||
      potential_commands_[0].size() != 2)
    PRINT_INPUT_ERROR(
      "PRD currently requires exactly one standard NEP potential command.");
  if (!is_supported_energy_nep_name(
        read_potential_name(potential_commands_[0][1])))
    PRINT_INPUT_ERROR(
      "PRD explicit streams support standard energy NEP4/NEP5 models.");
  potential_hash_ = hash_file(potential_commands_[0][1]);

  double final_temperature = 0.0;
  if (
    ensemble_command_.size() != 5 ||
    ensemble_command_[0] != "ensemble" ||
    ensemble_command_[1] != "nvt_bdp" ||
    !is_valid_real(ensemble_command_[2].c_str(), &temperature_) ||
    !is_valid_real(
      ensemble_command_[3].c_str(), &final_temperature) ||
    temperature_ <= 0.0 || final_temperature <= 0.0 ||
    !values_match(temperature_, final_temperature) ||
    !is_valid_real(
      ensemble_command_[4].c_str(), &temperature_coupling_) ||
    temperature_coupling_ < 1.0)
    PRINT_INPUT_ERROR(
      "PRD requires ensemble nvt_bdp T T tau with fixed positive T.");

  int number_of_devices = 0;
  CHECK(gpuGetDeviceCount(&number_of_devices));
  if (number_of_devices <= 0)
    PRINT_INPUT_ERROR("No GPU device is available for PRD.");
  if (config_.replicas_per_gpu == 0) {
    config_.replicas_per_gpu =
      (config_.replicas + number_of_devices - 1) / number_of_devices;
  }
  if (
    config_.replicas_per_gpu * number_of_devices <
    config_.replicas)
    PRINT_INPUT_ERROR(
      "PRD replicas_per_gpu is too small for the visible GPUs.");

  const size_t position_size =
    static_cast<size_t>(source_atom_.number_of_atoms) * 3;
  basin_position_.resize(position_size);
  hot_position_.resize(position_size);
  slot_data_.resize(config_.replicas);
  restart_bdp_states_.resize(config_.replicas);
  if (config_.resume) {
    read_restart_metadata();
    validate_append_event_file("prd_events.out");
    if (config_.dump_xyz) {
      validate_append_trajectory_file(
        config_.dump_prefix, source_atom_.number_of_atoms);
    }
  } else {
    selection_rng_.seed(config_.internal_seed + 1009U);
    dephase_rng_.seed(config_.internal_seed + 2003U);
  }

  printf("Parallel replica dynamics.\n");
  printf("    replicas: %d.\n", config_.replicas);
  if (config_.replicas == 1)
    printf("    warning: one PRD replica provides no clock acceleration.\n");
  printf("    replicas per GPU: %d.\n", config_.replicas_per_gpu);
  printf("    event check interval: %d steps.\n", config_.event_interval);
  if (config_.event_interval > 1)
    printf(
      "    warning: transient exits between event checks can be missed; "
      "use event_interval 1 for step-resolved detection and test "
      "interval convergence.\n");
  printf(
    "    dephasing: %d iterations of %d steps.\n",
    config_.dephase_iterations,
    config_.dephase_steps);
  if (config_.dephase_iterations == 0)
    printf(
      "    warning: dephasing is disabled; replicas are not guaranteed "
      "to sample the QSD.\n");
  printf(
    "    correlation window: %d steps.\n",
    config_.correlation_steps);
  printf(
    "    event displacement: %g A.\n",
    config_.event_distance);
  if (!source_box_.pbc_x || !source_box_.pbc_y || !source_box_.pbc_z)
    printf(
      "    warning: the event criterion is not invariant to rigid motion "
      "in nonperiodic systems.\n");
  printf(
    "    FIRE quench: f_max < %g eV/A within %d steps.\n",
    config_.quench_force_tolerance,
    config_.quench_steps);
  printf(
    "    thermostat: nvt_bdp at %g K with tau_T = %g steps.\n",
    temperature_,
    temperature_coupling_);
  if (config_.resume) {
    printf(
      "    restart prefix: %s at dynamics step %lld and clock %lld.\n",
      config_.resume_prefix.c_str(),
      completed_dynamics_steps_,
      clock_);
  }
  if (config_.dump_xyz) {
    printf(
      "    event trajectory: %s (dump interval is ignored).\n",
      config_.dump_prefix.c_str());
  }
  fflush(stdout);
}

void PRD_Driver::initialize_slot(Replica_Slot& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  if (config_.resume)
    read_restart_xyz(slot);
  allocate_memory_gpu(slot.group, slot.atom, slot.thermo);
  if (config_.resume) {
    slot.atom.velocity_per_atom.copy_from_host(
      slot.atom.cpu_velocity_per_atom.data());
  } else {
    Velocity velocity;
    velocity.initialize(
      false,
      temperature_,
      slot.atom,
      true,
      make_positive_seed(
        config_.internal_seed + 104729U * (slot.replica_id + 1U)));
  }

  slot.target_temperature = temperature_;
  double move_velocity[3] = {0.0, 0.0, 0.0};
  slot.thermostat.reset(new Ensemble_BDP(
    4, -1, move_velocity, temperature_, temperature_coupling_));
  if (config_.resume) {
    if (!slot.thermostat->import_rng_state(
          restart_bdp_states_[slot.replica_id]))
      PRINT_INPUT_ERROR("Invalid nvt_bdp RNG state in the PRD restart.");
  } else {
    slot.thermostat->seed_rng(
      config_.internal_seed + 130363U * (slot.replica_id + 1U) + 17U);
  }

  slot_data_[slot.replica_id].reset(new PRD_Slot_Data(
    slot,
    config_.quench_steps,
    config_.quench_force_tolerance));
}

void PRD_Driver::initialize_runtime()
{
  runtime_.initialize(
    config_.replicas,
    config_.replicas_per_gpu,
    [this](Replica_Slot& slot) { initialize_slot(slot); });
  runtime_.start_workers();
}

void PRD_Driver::open_output()
{
  const bool append_events =
    config_.resume && file_has_content("prd_events.out");
  event_file_ = my_fopen(
    "prd_events.out", config_.resume ? "a" : "w");
  if (!append_events) {
    fprintf(event_file_, "%s\n", event_output_marker);
    fprintf(
      event_file_,
      "# dynamics_step clock_step physical_time_fs event correlated "
      "coincident replica search_steps max_displacement_A "
      "quenched_potential_eV quenched_fmax_eV_per_A quench_steps\n");
  }
  fflush(event_file_);

  if (config_.dump_xyz) {
    trajectory_had_content_ =
      config_.resume && file_has_content(config_.dump_prefix);
    trajectory_file_ = my_fopen(
      config_.dump_prefix.c_str(), config_.resume ? "a" : "w");
  }
}

void PRD_Driver::close_output()
{
  if (event_file_) {
    fclose(event_file_);
    event_file_ = nullptr;
  }
  if (trajectory_file_) {
    fclose(trajectory_file_);
    trajectory_file_ = nullptr;
  }
}

long long PRD_Driver::global_dynamics_step() const
{
  return completed_dynamics_steps_ + dynamics_steps_;
}

void PRD_Driver::save_hot(const std::vector<int>& replica_ids)
{
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    slot_data_[replica_id]->hot.save(slot);
  }
}

void PRD_Driver::restore_hot(const std::vector<int>& replica_ids)
{
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    slot_data_[replica_id]->hot.restore(slot);
  }
  runtime_.compute_forces_selected(replica_ids);
}

void PRD_Driver::save_block_start(
  const std::vector<int>& replica_ids)
{
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    PRD_Slot_Data& data = *slot_data_[replica_id];
    data.block_start.save(slot);
    data.block_rng_state = bdp(replica_id).get_rng_state();
  }
}

void PRD_Driver::restore_block_start(
  const std::vector<int>& replica_ids)
{
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    PRD_Slot_Data& data = *slot_data_[replica_id];
    data.block_start.restore(slot);
    bdp(replica_id).set_rng_state(data.block_rng_state);
  }
  runtime_.compute_forces_selected(replica_ids);
}

void PRD_Driver::quench(const std::vector<int>& replica_ids)
{
  std::vector<std::thread> threads;
  threads.reserve(replica_ids.size());
  for (const int replica_id : replica_ids) {
    threads.emplace_back([this, replica_id] {
      Replica_Slot& slot = *runtime_.slots()[replica_id];
      CHECK(gpuSetDevice(slot.device_id));
      PRD_Slot_Data& data = *slot_data_[replica_id];
      data.quench_result = data.minimizer.compute(
        runtime_.force(slot),
        slot.box,
        slot.atom,
        slot.atom.position_per_atom,
        slot.group,
        slot.stream.get(),
        false);
    });
  }
  for (std::thread& thread : threads)
    thread.join();

  for (const int replica_id : replica_ids) {
    const Minimization_Result& result =
      slot_data_[replica_id]->quench_result;
    if (!result.converged) {
      const std::string message =
        "PRD FIRE quench did not converge for replica " +
        std::to_string(replica_id) + ".";
      PRINT_INPUT_ERROR(message.c_str());
    }
  }
}

std::vector<int> PRD_Driver::detect_events(
  const std::vector<int>& replica_ids)
{
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    PRD_Slot_Data& data = *slot_data_[replica_id];
    CHECK(gpuSetDevice(slot.device_id));
    gpu_maximum_displacement_squared<<<1, 256, 0, slot.stream.get()>>>(
      slot.atom.number_of_atoms,
      slot.box,
      slot.atom.position_per_atom.data(),
      data.basin.position().data(),
      data.maximum_displacement_squared.data());
    GPU_CHECK_KERNEL
    data.maximum_displacement_squared.copy_to_host_async(
      data.host_maximum_displacement_squared.data(),
      slot.stream.get());
  }

  std::vector<int> events;
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    PRD_Slot_Data& data = *slot_data_[replica_id];
    slot.stream.synchronize();
    const double maximum_squared =
      data.host_maximum_displacement_squared[0];
    if (!std::isfinite(maximum_squared) || maximum_squared < 0.0)
      PRINT_INPUT_ERROR(
        "PRD encountered an invalid event displacement.");
    data.maximum_displacement = std::sqrt(maximum_squared);
    if (data.maximum_displacement >= config_.event_distance)
      events.push_back(replica_id);
  }
  return events;
}

void PRD_Driver::broadcast_basin(const int replica_id)
{
  Replica_Slot& source = *runtime_.slots()[replica_id];
  slot_data_[replica_id]->basin.copy_position_to_host(
    basin_position_, source);
  for (int destination = 0; destination < config_.replicas; ++destination) {
    Replica_Slot& slot = *runtime_.slots()[destination];
    slot_data_[destination]->basin.copy_position_from_host(
      basin_position_, slot);
  }
  for (Replica_Slot* slot : runtime_.slots())
    slot->stream.synchronize();
}

void PRD_Driver::broadcast_hot_position(const int replica_id)
{
  Replica_Slot& source = *runtime_.slots()[replica_id];
  CHECK(gpuSetDevice(source.device_id));
  source.atom.position_per_atom.copy_to_host_async(
    hot_position_.data(), source.stream.get());
  source.stream.synchronize();

  for (Replica_Slot* slot : runtime_.slots()) {
    CHECK(gpuSetDevice(slot->device_id));
    slot->atom.position_per_atom.copy_from_host_async(
      hot_position_.data(), slot->stream.get());
  }
  for (Replica_Slot* slot : runtime_.slots())
    slot->stream.synchronize();
  runtime_.compute_forces_all();
}

void PRD_Driver::initialize_basin()
{
  if (config_.resume) {
    read_restart_basin();
    for (int replica_id = 0; replica_id < config_.replicas; ++replica_id) {
      Replica_Slot& slot = *runtime_.slots()[replica_id];
      slot_data_[replica_id]->basin.copy_position_from_host(
        basin_position_, slot);
    }
    for (Replica_Slot* slot : runtime_.slots())
      slot->stream.synchronize();
    if (trajectory_file_ && !trajectory_had_content_)
      write_event_trajectory(-1, false);
    printf("    Restored the PRD basin reference.\n");
    fflush(stdout);
    return;
  }

  const std::vector<int> initial_replica(1, 0);
  save_hot(initial_replica);
  quench(initial_replica);
  Replica_Slot& slot = *runtime_.slots()[0];
  slot_data_[0]->basin.save(slot);
  broadcast_basin(0);
  write_event_trajectory(-1, false);
  restore_hot(initial_replica);
  printf(
    "    Initial basin: potential = %.10f eV, f_max = %.6e eV/A.\n",
    slot_data_[0]->quench_result.total_potential,
    slot_data_[0]->quench_result.force_max);
  fflush(stdout);
}

void PRD_Driver::randomize_velocities(
  const std::vector<int>& replica_ids)
{
  std::uniform_int_distribution<int> seed_distribution(
    1, std::numeric_limits<int>::max());
  for (const int replica_id : replica_ids) {
    Replica_Slot& slot = *runtime_.slots()[replica_id];
    CHECK(gpuSetDevice(slot.device_id));
    slot.atom.position_per_atom.copy_to_host(
      slot.atom.cpu_position_per_atom.data());
    Velocity velocity;
    velocity.initialize(
      false,
      temperature_,
      slot.atom,
      true,
      seed_distribution(dephase_rng_));
  }
}

void PRD_Driver::dephase()
{
  if (config_.dephase_iterations == 0)
    return;
  const std::vector<int> replicas = all_replicas();

  for (int iteration = 0;
       iteration < config_.dephase_iterations;
       ++iteration) {
    for (const int replica_id : replicas) {
      Replica_Slot& slot = *runtime_.slots()[replica_id];
      slot_data_[replica_id]->dephase_start.save(slot);
    }
    for (const int replica_id : replicas)
      runtime_.slots()[replica_id]->stream.synchronize();

    std::vector<int> pending = replicas;
    int retries = 0;
    while (!pending.empty()) {
      if (++retries > config_.maximum_dephase_retries)
        PRINT_INPUT_ERROR(
          "PRD dephasing exceeded max_dephase_retries.");
      randomize_velocities(pending);
      std::vector<int> active = pending;
      std::vector<int> retry;
      int completed_steps = 0;
      while (!active.empty() && completed_steps < config_.dephase_steps) {
        const int block_steps = std::min(
          config_.event_interval,
          config_.dephase_steps - completed_steps);
        runtime_.advance_selected(active, block_steps);
        save_hot(active);
        quench(active);
        const std::vector<int> escaped = detect_events(active);
        std::vector<unsigned char> has_escaped(config_.replicas, 0);
        for (const int replica_id : escaped)
          has_escaped[replica_id] = 1;

        std::vector<int> survivors;
        survivors.reserve(active.size());
        for (const int replica_id : active) {
          Replica_Slot& slot = *runtime_.slots()[replica_id];
          if (has_escaped[replica_id]) {
            slot_data_[replica_id]->dephase_start.restore(slot);
            retry.push_back(replica_id);
          } else {
            slot_data_[replica_id]->hot.restore(slot);
            survivors.push_back(replica_id);
          }
        }
        runtime_.compute_forces_selected(active);
        dephase_rejections_ += static_cast<int>(escaped.size());
        active.swap(survivors);
        completed_steps += block_steps;
      }
      pending.swap(retry);
    }
  }
}

Search_Result PRD_Driver::search_block(
  const std::vector<int>& replica_ids,
  const int maximum_steps)
{
  if (replica_ids.empty() || maximum_steps <= 0)
    PRINT_INPUT_ERROR("Invalid PRD event-search block.");

  save_block_start(replica_ids);
  runtime_.advance_selected(replica_ids, maximum_steps);
  save_hot(replica_ids);
  quench(replica_ids);
  std::vector<int> candidates = detect_events(replica_ids);

  Search_Result result;
  result.steps = maximum_steps;
  if (candidates.empty()) {
    restore_hot(replica_ids);
    return result;
  }

  // Once an event is present at a check point, replay the block to locate its
  // first detected step. Exits that return before the check point are below
  // the configured event_interval resolution.
  restore_block_start(replica_ids);
  for (int step = 0; step < maximum_steps; ++step) {
    runtime_.advance_selected(replica_ids, 1);
    save_hot(replica_ids);
    quench(replica_ids);
    const std::vector<int> events = detect_events(replica_ids);
    if (!events.empty()) {
      std::uniform_int_distribution<size_t> choose(
        0, events.size() - 1);
      result.event = true;
      result.winner = events[choose(selection_rng_)];
      result.steps = step + 1;
      result.coincident = static_cast<int>(events.size());
      result.maximum_displacement =
        slot_data_[result.winner]->maximum_displacement;
      result.quench = slot_data_[result.winner]->quench_result;
      return result;
    }
    restore_hot(replica_ids);
  }

  PRINT_INPUT_ERROR(
    "PRD event replay did not reproduce the block-end event.");
  return result;
}

void PRD_Driver::write_event(
  const Search_Result& result,
  const bool correlated)
{
  if (!event_file_)
    return;
  fprintf(
    event_file_,
    "%lld %lld %.16g %d %d %d %d %d %.16g %.16g %.16g %d\n",
    global_dynamics_step(),
    clock_,
    clock_ * time_step_ * TIME_UNIT_CONVERSION,
    event_number_,
    correlated ? 1 : 0,
    result.coincident,
    result.winner,
    result.steps,
    result.maximum_displacement,
    result.quench.total_potential,
    result.quench.force_max,
    result.quench.steps);
  fflush(event_file_);
}

void PRD_Driver::write_event_trajectory(
  const int replica_id, const bool correlated)
{
  if (!trajectory_file_)
    return;
  const int number_of_atoms = source_atom_.number_of_atoms;
  fprintf(trajectory_file_, "%d\n", number_of_atoms);
  fprintf(
    trajectory_file_,
    "Time=%.8f dynamics_step=%lld clock_step=%lld event=%d "
    "correlated=%d replica=%d initial=%d replica_mode=prd_event ",
    clock_ * time_step_ * TIME_UNIT_CONVERSION,
    global_dynamics_step(),
    clock_,
    event_number_,
    correlated ? 1 : 0,
    replica_id,
    replica_id < 0 ? 1 : 0);
  fprintf(
    trajectory_file_,
    "pbc=\"%c %c %c\" ",
    source_box_.pbc_x ? 'T' : 'F',
    source_box_.pbc_y ? 'T' : 'F',
    source_box_.pbc_z ? 'T' : 'F');
  fprintf(
    trajectory_file_,
    "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" "
    "Properties=species:S:1:pos:R:3\n",
    source_box_.cpu_h[0],
    source_box_.cpu_h[3],
    source_box_.cpu_h[6],
    source_box_.cpu_h[1],
    source_box_.cpu_h[4],
    source_box_.cpu_h[7],
    source_box_.cpu_h[2],
    source_box_.cpu_h[5],
    source_box_.cpu_h[8]);
  for (int atom = 0; atom < number_of_atoms; ++atom) {
    fprintf(
      trajectory_file_,
      "%s %.17g %.17g %.17g\n",
      source_atom_.cpu_atom_symbol[atom].c_str(),
      basin_position_[atom],
      basin_position_[atom + number_of_atoms],
      basin_position_[atom + 2 * number_of_atoms]);
  }
  fflush(trajectory_file_);
}

void PRD_Driver::accept_event(
  const Search_Result& result,
  const bool correlated)
{
  Replica_Slot& winner = *runtime_.slots()[result.winner];
  slot_data_[result.winner]->basin.save(winner);
  broadcast_basin(result.winner);
  ++event_number_;
  write_event(result, correlated);
  write_event_trajectory(result.winner, correlated);
  slot_data_[result.winner]->hot.restore(winner);
  runtime_.compute_forces_selected(
    std::vector<int>(1, result.winner));

  printf(
    "    PRD event %d: step %lld, clock %lld, replica %d%s, "
    "coincident %d.\n",
    event_number_,
    global_dynamics_step(),
    clock_,
    result.winner,
    correlated ? " (correlated)" : "",
    result.coincident);
  fflush(stdout);
}

void PRD_Driver::finish_correlation()
{
  if (
    phase_ != PRD_Phase::correlation || correlation_winner_ < 0 ||
    correlation_winner_ >= config_.replicas)
    PRINT_INPUT_ERROR("Invalid PRD correlation state.");
  broadcast_hot_position(correlation_winner_);
  phase_ = PRD_Phase::dephase;
  correlation_winner_ = -1;
  correlation_quiet_steps_ = 0;
}

void PRD_Driver::write_restart_xyz(Replica_Slot& slot) const
{
  CHECK(gpuSetDevice(slot.device_id));
  slot.stream.synchronize();
  slot.atom.position_per_atom.copy_to_host(
    slot.atom.cpu_position_per_atom.data());
  slot.atom.velocity_per_atom.copy_to_host(
    slot.atom.cpu_velocity_per_atom.data());

  const std::string filename =
    "prd_replica_" + std::to_string(slot.replica_id) + "_restart.xyz";
  FILE* file = my_fopen(filename.c_str(), "w");
  const int number_of_atoms = slot.atom.number_of_atoms;
  fprintf(file, "%d\n", number_of_atoms);
  fprintf(
    file,
    "pbc=\"%c %c %c\" ",
    slot.box.pbc_x ? 'T' : 'F',
    slot.box.pbc_y ? 'T' : 'F',
    slot.box.pbc_z ? 'T' : 'F');
  fprintf(
    file,
    "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" ",
    slot.box.cpu_h[0],
    slot.box.cpu_h[3],
    slot.box.cpu_h[6],
    slot.box.cpu_h[1],
    slot.box.cpu_h[4],
    slot.box.cpu_h[7],
    slot.box.cpu_h[2],
    slot.box.cpu_h[5],
    slot.box.cpu_h[8]);
  fprintf(
    file,
    "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 "
    "replica_mode=prd replica=%d dynamics_step=%lld clock_step=%lld\n",
    slot.replica_id,
    global_dynamics_step(),
    clock_);
  for (int atom = 0; atom < number_of_atoms; ++atom) {
    fprintf(
      file,
      "%s %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
      source_atom_.cpu_atom_symbol[atom].c_str(),
      slot.atom.cpu_position_per_atom[atom],
      slot.atom.cpu_position_per_atom[atom + number_of_atoms],
      slot.atom.cpu_position_per_atom[atom + 2 * number_of_atoms],
      source_atom_.cpu_mass[atom],
      slot.atom.cpu_velocity_per_atom[atom] / TIME_UNIT_CONVERSION,
      slot.atom.cpu_velocity_per_atom[atom + number_of_atoms] /
        TIME_UNIT_CONVERSION,
      slot.atom.cpu_velocity_per_atom[atom + 2 * number_of_atoms] /
        TIME_UNIT_CONVERSION);
  }
  fclose(file);
}

void PRD_Driver::write_restart_basin() const
{
  FILE* file = my_fopen("prd_basin_restart.xyz", "w");
  const int number_of_atoms = source_atom_.number_of_atoms;
  fprintf(file, "%d\n", number_of_atoms);
  fprintf(
    file,
    "pbc=\"%c %c %c\" ",
    source_box_.pbc_x ? 'T' : 'F',
    source_box_.pbc_y ? 'T' : 'F',
    source_box_.pbc_z ? 'T' : 'F');
  fprintf(
    file,
    "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" ",
    source_box_.cpu_h[0],
    source_box_.cpu_h[3],
    source_box_.cpu_h[6],
    source_box_.cpu_h[1],
    source_box_.cpu_h[4],
    source_box_.cpu_h[7],
    source_box_.cpu_h[2],
    source_box_.cpu_h[5],
    source_box_.cpu_h[8]);
  fprintf(
    file,
    "Properties=species:S:1:pos:R:3:mass:R:1 replica_mode=prd_basin "
    "dynamics_step=%lld clock_step=%lld\n",
    global_dynamics_step(),
    clock_);
  for (int atom = 0; atom < number_of_atoms; ++atom) {
    fprintf(
      file,
      "%s %.17g %.17g %.17g %.17g\n",
      source_atom_.cpu_atom_symbol[atom].c_str(),
      basin_position_[atom],
      basin_position_[atom + number_of_atoms],
      basin_position_[atom + 2 * number_of_atoms],
      source_atom_.cpu_mass[atom]);
  }
  fclose(file);
}

void PRD_Driver::write_restart_metadata() const
{
  FILE* file = my_fopen("prd_restart.meta", "w");
  fprintf(file, "# GPUMD PRD restart metadata\n");
  fprintf(file, "format_version %d\n", restart_format_version);
  fprintf(file, "mode prd\n");
  fprintf(file, "replicas %d\n", config_.replicas);
  fprintf(
    file,
    "completed_dynamics_steps %lld\n",
    global_dynamics_step());
  fprintf(file, "clock %lld\n", clock_);
  fprintf(file, "event_number %d\n", event_number_);
  fprintf(file, "dephase_rejections %d\n", dephase_rejections_);
  fprintf(file, "phase %s\n", phase_name(phase_));
  fprintf(file, "correlation_winner %d\n", correlation_winner_);
  fprintf(
    file,
    "correlation_quiet_steps %d\n",
    correlation_quiet_steps_);
  fprintf(file, "time_step %.17g\n", time_step_);
  fprintf(file, "temperature %.17g\n", temperature_);
  fprintf(
    file,
    "temperature_coupling %.17g\n",
    temperature_coupling_);
  fprintf(file, "potential_hash %llu\n", potential_hash_);
  fprintf(file, "event_interval %d\n", config_.event_interval);
  fprintf(file, "dephase_iterations %d\n", config_.dephase_iterations);
  fprintf(file, "dephase_steps %d\n", config_.dephase_steps);
  fprintf(file, "correlation_steps %d\n", config_.correlation_steps);
  fprintf(file, "event_distance %.17g\n", config_.event_distance);
  fprintf(
    file,
    "quench_force_tolerance %.17g\n",
    config_.quench_force_tolerance);
  fprintf(file, "quench_steps %d\n", config_.quench_steps);
  fprintf(
    file,
    "max_dephase_retries %d\n",
    config_.maximum_dephase_retries);
  std::ostringstream selection_state;
  selection_state << selection_rng_;
  fprintf(file, "selection_rng %s\n", selection_state.str().c_str());
  std::ostringstream dephase_state;
  dephase_state << dephase_rng_;
  fprintf(file, "dephase_rng %s\n", dephase_state.str().c_str());
  for (const Replica_Slot* slot : runtime_.slots()) {
    fprintf(
      file,
      "bdp_rng %d %s\n",
      slot->replica_id,
      slot->thermostat->export_rng_state().c_str());
  }
  fclose(file);
}

void PRD_Driver::write_restart()
{
  for (Replica_Slot* slot : runtime_.slots())
    write_restart_xyz(*slot);
  write_restart_basin();
  write_restart_metadata();
}

void PRD_Driver::read_restart_xyz(Replica_Slot& slot)
{
  const std::string filename =
    config_.resume_prefix + "_replica_" +
    std::to_string(slot.replica_id) + "_restart.xyz";
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open a PRD replica restart xyz file.");
  int number_of_atoms = 0;
  if (!(input >> number_of_atoms) ||
      number_of_atoms != source_atom_.number_of_atoms)
    PRINT_INPUT_ERROR(
      "PRD restart atom count does not match model.xyz.");
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string comment;
  if (!std::getline(input, comment))
    PRINT_INPUT_ERROR("Cannot read the PRD restart xyz comment line.");
  validate_xyz_box(comment, source_box_, "PRD");

  std::string value;
  int file_replica = -1;
  long long dynamics_step = -1;
  long long clock_step = -1;
  if (!extract_token_value(comment, "replica_mode", value) ||
      value != "prd" ||
      !extract_token_value(comment, "replica", value) ||
      !parse_scalar(value, file_replica) ||
      file_replica != slot.replica_id ||
      !extract_token_value(comment, "Properties", value) ||
      value != "species:S:1:pos:R:3:mass:R:1:vel:R:3" ||
      !extract_token_value(comment, "dynamics_step", value) ||
      !parse_scalar(value, dynamics_step) ||
      dynamics_step != completed_dynamics_steps_ ||
      !extract_token_value(comment, "clock_step", value) ||
      !parse_scalar(value, clock_step) || clock_step != clock_)
    PRINT_INPUT_ERROR("PRD restart xyz metadata is invalid.");

  const int n = source_atom_.number_of_atoms;
  for (int atom = 0; atom < n; ++atom) {
    std::string symbol;
    double mass = 0.0;
    if (!(input >> symbol >> slot.atom.cpu_position_per_atom[atom] >>
          slot.atom.cpu_position_per_atom[atom + n] >>
          slot.atom.cpu_position_per_atom[atom + 2 * n] >> mass >>
          slot.atom.cpu_velocity_per_atom[atom] >>
          slot.atom.cpu_velocity_per_atom[atom + n] >>
          slot.atom.cpu_velocity_per_atom[atom + 2 * n]))
      PRINT_INPUT_ERROR("PRD restart xyz atom data is incomplete.");
    if (symbol != source_atom_.cpu_atom_symbol[atom] ||
        !values_match(mass, source_atom_.cpu_mass[atom]))
      PRINT_INPUT_ERROR(
        "PRD restart xyz species or mass does not match model.xyz.");
    for (int dimension = 0; dimension < 3; ++dimension) {
      const int index = atom + dimension * n;
      if (!std::isfinite(slot.atom.cpu_position_per_atom[index]) ||
          !std::isfinite(slot.atom.cpu_velocity_per_atom[index]))
        PRINT_INPUT_ERROR(
          "PRD restart xyz contains a non-finite atom value.");
      slot.atom.cpu_velocity_per_atom[index] *= TIME_UNIT_CONVERSION;
    }
  }
}

void PRD_Driver::read_restart_basin()
{
  const std::string filename =
    config_.resume_prefix + "_basin_restart.xyz";
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the PRD basin restart xyz file.");
  int number_of_atoms = 0;
  if (!(input >> number_of_atoms) ||
      number_of_atoms != source_atom_.number_of_atoms)
    PRINT_INPUT_ERROR(
      "PRD basin restart atom count does not match model.xyz.");
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string comment;
  if (!std::getline(input, comment))
    PRINT_INPUT_ERROR(
      "Cannot read the PRD basin restart xyz comment line.");
  validate_xyz_box(comment, source_box_, "PRD basin");

  std::string value;
  long long dynamics_step = -1;
  long long clock_step = -1;
  if (!extract_token_value(comment, "replica_mode", value) ||
      value != "prd_basin" ||
      !extract_token_value(comment, "Properties", value) ||
      value != "species:S:1:pos:R:3:mass:R:1" ||
      !extract_token_value(comment, "dynamics_step", value) ||
      !parse_scalar(value, dynamics_step) ||
      dynamics_step != completed_dynamics_steps_ ||
      !extract_token_value(comment, "clock_step", value) ||
      !parse_scalar(value, clock_step) || clock_step != clock_)
    PRINT_INPUT_ERROR("PRD basin restart xyz metadata is invalid.");

  const int n = source_atom_.number_of_atoms;
  for (int atom = 0; atom < n; ++atom) {
    std::string symbol;
    double mass = 0.0;
    if (!(input >> symbol >> basin_position_[atom] >>
          basin_position_[atom + n] >>
          basin_position_[atom + 2 * n] >> mass))
      PRINT_INPUT_ERROR(
        "PRD basin restart xyz atom data is incomplete.");
    if (symbol != source_atom_.cpu_atom_symbol[atom] ||
        !values_match(mass, source_atom_.cpu_mass[atom]))
      PRINT_INPUT_ERROR(
        "PRD basin restart species or mass does not match model.xyz.");
    for (int dimension = 0; dimension < 3; ++dimension) {
      if (!std::isfinite(basin_position_[atom + dimension * n]))
        PRINT_INPUT_ERROR(
          "PRD basin restart contains a non-finite position.");
    }
  }
}

void PRD_Driver::read_restart_metadata()
{
  const std::string filename = config_.resume_prefix + "_restart.meta";
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the PRD restart metadata file.");
  std::string header;
  if (!std::getline(input, header) ||
      header != "# GPUMD PRD restart metadata")
    PRINT_INPUT_ERROR("Invalid PRD restart metadata header.");

  int version = 0;
  read_restart_value(input, "format_version", version);
  if (version != restart_format_version)
    PRINT_INPUT_ERROR("Unsupported PRD restart format_version.");
  std::string mode;
  read_restart_value(input, "mode", mode);
  if (mode != "prd")
    PRINT_INPUT_ERROR("PRD restart mode is invalid.");
  int replicas = 0;
  read_restart_value(input, "replicas", replicas);
  if (replicas != config_.replicas)
    PRINT_INPUT_ERROR("PRD restart replicas do not match the input.");

  read_restart_value(
    input, "completed_dynamics_steps", completed_dynamics_steps_);
  read_restart_value(input, "clock", clock_);
  read_restart_value(input, "event_number", event_number_);
  read_restart_value(
    input, "dephase_rejections", dephase_rejections_);
  if (
    completed_dynamics_steps_ < 0 || clock_ < completed_dynamics_steps_ ||
    event_number_ < 0 || dephase_rejections_ < 0)
    PRINT_INPUT_ERROR("PRD restart counters are invalid.");

  std::string stored_phase;
  read_restart_value(input, "phase", stored_phase);
  if (stored_phase == "dephase")
    phase_ = PRD_Phase::dephase;
  else if (stored_phase == "parallel")
    phase_ = PRD_Phase::parallel;
  else if (stored_phase == "correlation")
    phase_ = PRD_Phase::correlation;
  else
    PRINT_INPUT_ERROR("PRD restart phase is invalid.");
  read_restart_value(
    input, "correlation_winner", correlation_winner_);
  read_restart_value(
    input, "correlation_quiet_steps", correlation_quiet_steps_);

  double value = 0.0;
  read_restart_value(input, "time_step", value);
  if (!values_match(value, time_step_))
    PRINT_INPUT_ERROR("PRD restart time_step does not match the input.");
  read_restart_value(input, "temperature", value);
  if (!values_match(value, temperature_))
    PRINT_INPUT_ERROR("PRD restart temperature does not match the input.");
  read_restart_value(input, "temperature_coupling", value);
  if (!values_match(value, temperature_coupling_))
    PRINT_INPUT_ERROR(
      "PRD restart nvt_bdp coupling does not match the input.");
  unsigned long long stored_hash = 0;
  read_restart_value(input, "potential_hash", stored_hash);
  if (stored_hash != potential_hash_)
    PRINT_INPUT_ERROR("PRD restart potential file does not match the input.");

  int integer_value = 0;
  read_restart_value(input, "event_interval", integer_value);
  if (integer_value != config_.event_interval)
    PRINT_INPUT_ERROR("PRD restart event_interval does not match the input.");
  read_restart_value(input, "dephase_iterations", integer_value);
  if (integer_value != config_.dephase_iterations)
    PRINT_INPUT_ERROR(
      "PRD restart dephase_iterations does not match the input.");
  read_restart_value(input, "dephase_steps", integer_value);
  if (integer_value != config_.dephase_steps)
    PRINT_INPUT_ERROR("PRD restart dephase_steps does not match the input.");
  read_restart_value(input, "correlation_steps", integer_value);
  if (integer_value != config_.correlation_steps)
    PRINT_INPUT_ERROR(
      "PRD restart correlation_steps does not match the input.");
  read_restart_value(input, "event_distance", value);
  if (!values_match(value, config_.event_distance))
    PRINT_INPUT_ERROR("PRD restart event_distance does not match the input.");
  read_restart_value(input, "quench_force_tolerance", value);
  if (!values_match(value, config_.quench_force_tolerance))
    PRINT_INPUT_ERROR(
      "PRD restart quench_force_tolerance does not match the input.");
  read_restart_value(input, "quench_steps", integer_value);
  if (integer_value != config_.quench_steps)
    PRINT_INPUT_ERROR("PRD restart quench_steps does not match the input.");
  read_restart_value(input, "max_dephase_retries", integer_value);
  if (integer_value != config_.maximum_dephase_retries)
    PRINT_INPUT_ERROR(
      "PRD restart max_dephase_retries does not match the input.");

  std::istringstream selection_state(
    read_restart_line(input, "selection_rng"));
  std::string trailing;
  if (!(selection_state >> selection_rng_) || selection_state >> trailing)
    PRINT_INPUT_ERROR("PRD restart selection_rng is invalid.");
  std::istringstream dephase_state(
    read_restart_line(input, "dephase_rng"));
  if (!(dephase_state >> dephase_rng_) || dephase_state >> trailing)
    PRINT_INPUT_ERROR("PRD restart dephase_rng is invalid.");

  for (int replica = 0; replica < config_.replicas; ++replica) {
    std::string key;
    int stored_replica = -1;
    if (!(input >> key >> stored_replica) || key != "bdp_rng" ||
        stored_replica != replica)
      PRINT_INPUT_ERROR("PRD restart bdp_rng replica id is invalid.");
    std::getline(input, restart_bdp_states_[replica]);
    restart_bdp_states_[replica] =
      trim_left(restart_bdp_states_[replica]);
    if (restart_bdp_states_[replica].empty())
      PRINT_INPUT_ERROR("PRD restart bdp_rng state is empty.");
  }
  std::string extra;
  if (input >> extra)
    PRINT_INPUT_ERROR("PRD restart metadata has unexpected trailing data.");

  if (phase_ == PRD_Phase::correlation) {
    if (
      config_.correlation_steps <= 0 || correlation_winner_ < 0 ||
      correlation_winner_ >= config_.replicas ||
      correlation_quiet_steps_ < 0 ||
      correlation_quiet_steps_ >= config_.correlation_steps)
      PRINT_INPUT_ERROR("PRD restart correlation state is invalid.");
  } else if (
    correlation_winner_ != -1 || correlation_quiet_steps_ != 0) {
    PRINT_INPUT_ERROR("PRD restart contains a stale correlation state.");
  }
}

void PRD_Driver::run()
{
  validate_input();
  initialize_runtime();
  open_output();
  initialize_basin();

  const auto time_begin = std::chrono::high_resolution_clock::now();
  const long long clock_begin = clock_;
  const std::vector<int> replicas = all_replicas();
  while (dynamics_steps_ < number_of_steps_) {
    if (phase_ == PRD_Phase::dephase) {
      dephase();
      phase_ = PRD_Phase::parallel;
      continue;
    }

    if (phase_ == PRD_Phase::parallel) {
      const int block_steps = std::min(
        config_.event_interval,
        number_of_steps_ - dynamics_steps_);
      Search_Result result = search_block(replicas, block_steps);
      dynamics_steps_ += result.steps;
      clock_ +=
        static_cast<long long>(result.steps) * config_.replicas;
      if (!result.event)
        continue;

      accept_event(result, false);
      phase_ = PRD_Phase::correlation;
      correlation_winner_ = result.winner;
      correlation_quiet_steps_ = 0;
      if (config_.correlation_steps == 0)
        finish_correlation();
      continue;
    }

    if (correlation_quiet_steps_ >= config_.correlation_steps) {
      finish_correlation();
      continue;
    }
    const int correlation_block = std::min(
      config_.event_interval,
      std::min(
        config_.correlation_steps - correlation_quiet_steps_,
        number_of_steps_ - dynamics_steps_));
    Search_Result correlated = search_block(
      std::vector<int>(1, correlation_winner_),
      correlation_block);
    dynamics_steps_ += correlated.steps;
    clock_ += correlated.steps;
    if (correlated.event) {
      accept_event(correlated, true);
      correlation_winner_ = correlated.winner;
      correlation_quiet_steps_ = 0;
    } else {
      correlation_quiet_steps_ += correlated.steps;
    }
    if (correlation_quiet_steps_ >= config_.correlation_steps)
      finish_correlation();
  }

  runtime_.stop_workers();
  write_restart();
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed = time_finish - time_begin;
  printf(
    "PRD finished: %d dynamics steps, %lld accelerated clock steps, "
    "%d events, %d rejected dephase trials.\n",
    dynamics_steps_,
    clock_,
    event_number_,
    dephase_rejections_);
  printf("Time used for PRD = %g second.\n", elapsed.count());
  if (elapsed.count() > 0.0) {
    printf(
      "Effective PRD speed = %g atom*clock-step/second.\n",
      source_atom_.number_of_atoms *
        (static_cast<double>(clock_ - clock_begin) / elapsed.count()));
  }
  fflush(stdout);
  close_output();
}

} // namespace

namespace replica
{

void run_prd(
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  const std::vector<std::string>& multi_replica_command,
  const std::vector<std::string>& dump_xyz_command,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  const double time_step,
  const int number_of_steps)
{
  PRD_Driver driver(
    potential_commands,
    ensemble_command,
    multi_replica_command,
    dump_xyz_command,
    source_atom,
    source_box,
    source_group,
    time_step,
    number_of_steps);
  driver.run();
}

} // namespace replica
