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
Temperature replica-exchange molecular dynamics. Each visible GPU owns one
potential model. Independent replica states on that GPU use separate streams.
------------------------------------------------------------------------------*/

#include "replica.cuh"
#include "replica_common.cuh"
#include "velocity.cuh"
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
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using replica::Replica_Runtime;
using replica::Replica_Slot;
using replica::extract_quoted_value;
using replica::extract_token_value;
using replica::file_has_content;
using replica::hash_file;
using replica::is_supported_charge_nep_name;
using replica::is_supported_replica_nep_name;
using replica::parse_scalar;
using replica::read_potential_name;
using replica::trim_left;
using replica::validate_xyz_box;
using replica::values_match;

constexpr double target_acceptance = 0.2;
constexpr const char* exchange_output_marker =
  "# GPUMD REMD exchange output format 1";
constexpr const char* status_output_marker =
  "# GPUMD REMD temperature status output format 1";
constexpr const char* thermo_output_marker =
  "# GPUMD REMD thermodynamic output format 1";

void validate_append_output_file(const std::string& filename, const char* marker)
{
  if (!file_has_content(filename))
    return;
  std::ifstream input(filename);
  std::string first_line;
  if (!std::getline(input, first_line) || first_line != marker) {
    const std::string message =
      "REMD cannot append to " + filename +
      " because its output format is incompatible. Move or remove the existing file "
      "before resuming.";
    PRINT_INPUT_ERROR(message.c_str());
  }
}

int next_periodic_event(
  const int local_step,
  const int final_local_step,
  const long long global_step,
  const int interval)
{
  const long long remainder = global_step % interval;
  const long long steps = remainder == 0 ? interval : interval - remainder;
  return static_cast<int>(
    std::min<long long>(final_local_step, static_cast<long long>(local_step) + steps));
}

struct REMD_Config {
  enum Temperature_Style { geometric, linear };

  int replicas = 0;
  int replicas_per_gpu = 0;
  int exchange_interval = 0;
  int equilibration_steps = 0;
  int dump_interval = 0;
  bool resume = false;
  bool dump_xyz = false;
  bool dump_velocity = false;
  Temperature_Style temperature_style = geometric;
  std::string resume_prefix;
  std::string dump_prefix;
  std::vector<double> temperatures;
  std::vector<double> betas;
  unsigned int internal_seed = 1;

  void parse(
    const std::vector<std::string>& command,
    const std::vector<std::string>& dump_command);
};

void read_temperature_file(const std::string& filename, std::vector<double>& temperatures)
{
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the REMD temperature file.");
  while (input.peek() != EOF) {
    const std::vector<std::string> tokens = get_tokens(input);
    for (const std::string& token : tokens) {
      if (token.empty() || token[0] == '#')
        break;
      double temperature = 0.0;
      if (!is_valid_real(token.c_str(), &temperature) || temperature <= 0.0)
        PRINT_INPUT_ERROR("REMD temperatures must be positive real numbers.");
      temperatures.push_back(temperature);
    }
  }
}

void REMD_Config::parse(
  const std::vector<std::string>& command,
  const std::vector<std::string>& dump_command)
{
  if (command.size() < 3 || command[0] != "multi_replica" || command[1] != "remd")
    PRINT_INPUT_ERROR("multi_replica currently supports only remd.");

  enum Temperature_Source { unset, file, range };
  Temperature_Source source = unset;
  double minimum_temperature = 0.0;
  double maximum_temperature = 0.0;
  for (size_t i = 2; i < command.size();) {
    if (command[i] == "replicas") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &replicas) || replicas <= 1)
        PRINT_INPUT_ERROR("replicas should be an integer larger than 1.");
      ++i;
      if (i < command.size()) {
        int value = 0;
        if (is_valid_int(command[i].c_str(), &value)) {
          if (value <= 0)
            PRINT_INPUT_ERROR("replicas_per_gpu should be positive.");
          replicas_per_gpu = value;
          ++i;
        }
      }
    } else if (command[i] == "exchange") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &exchange_interval) || exchange_interval <= 0)
        PRINT_INPUT_ERROR("exchange should be a positive integer.");
      ++i;
    } else if (
      command[i] == "equilibrate" || command[i] == "equilibration" ||
      command[i] == "equilibration_steps") {
      if (++i >= command.size() ||
          !is_valid_int(command[i].c_str(), &equilibration_steps) ||
          equilibration_steps < 0)
        PRINT_INPUT_ERROR("equilibration should be a non-negative integer.");
      ++i;
    } else if (command[i] == "temp") {
      if (source != unset)
        PRINT_INPUT_ERROR("Specify only one REMD temperature source.");
      if (++i >= command.size())
        PRINT_INPUT_ERROR("temp requires a file name or Tmin Tmax.");
      if (is_valid_real(command[i].c_str(), &minimum_temperature)) {
        source = range;
        if (++i >= command.size() ||
            !is_valid_real(command[i].c_str(), &maximum_temperature) ||
            minimum_temperature <= 0.0 || maximum_temperature <= minimum_temperature)
          PRINT_INPUT_ERROR("temp should satisfy 0 < Tmin < Tmax.");
        ++i;
      } else {
        source = file;
        read_temperature_file(command[i++], temperatures);
      }
    } else if (command[i] == "spacing") {
      if (++i >= command.size())
        PRINT_INPUT_ERROR("spacing requires geometric or linear.");
      if (command[i] == "geometric")
        temperature_style = geometric;
      else if (command[i] == "linear")
        temperature_style = linear;
      else
        PRINT_INPUT_ERROR("spacing should be geometric or linear.");
      ++i;
    } else if (command[i] == "verbose" || command[i] == "verbose_output") {
      // Retain compatibility with earlier REMD inputs. Diagnostics are always written.
      ++i;
    } else if (command[i] == "resume") {
      if (++i >= command.size())
        PRINT_INPUT_ERROR("resume requires a restart prefix.");
      resume = true;
      resume_prefix = command[i++];
    } else if (command[i] == "dump_xyz") {
      PRINT_INPUT_ERROR("Use the ordinary dump_xyz command outside multi_replica.");
    } else {
      PRINT_INPUT_ERROR("Unknown multi_replica option.");
    }
  }

  if (replicas <= 1 || exchange_interval <= 0 || source == unset)
    PRINT_INPUT_ERROR("REMD requires replicas, exchange, and temp options.");
  if (resume && equilibration_steps > 0)
    PRINT_INPUT_ERROR("REMD resume cannot be combined with equilibration.");
  if (source == range) {
    temperatures.resize(replicas);
    for (int r = 0; r < replicas; ++r) {
      const double fraction = static_cast<double>(r) / (replicas - 1);
      temperatures[r] = temperature_style == linear
                          ? minimum_temperature +
                              fraction * (maximum_temperature - minimum_temperature)
                          : minimum_temperature *
                              std::pow(maximum_temperature / minimum_temperature, fraction);
    }
  }
  if (static_cast<int>(temperatures.size()) != replicas)
    PRINT_INPUT_ERROR("REMD requires exactly one temperature for each replica.");
  for (int r = 1; r < replicas; ++r) {
    if (!(temperatures[r] > temperatures[r - 1]))
      PRINT_INPUT_ERROR("REMD temperatures should be strictly increasing.");
  }
  betas.resize(replicas);
  for (int r = 0; r < replicas; ++r)
    betas[r] = 1.0 / (K_B * temperatures[r]);

  if (!dump_command.empty()) {
    if (dump_command.size() < 5 || dump_command[0] != "dump_xyz")
      PRINT_INPUT_ERROR("Invalid REMD dump_xyz command.");
    int grouping_method = 0;
    int group_id = 0;
    if (!is_valid_int(dump_command[1].c_str(), &grouping_method) ||
        grouping_method != -1 ||
        !is_valid_int(dump_command[2].c_str(), &group_id))
      PRINT_INPUT_ERROR("REMD dump_xyz requires grouping_method -1.");
    if (!is_valid_int(dump_command[3].c_str(), &dump_interval) || dump_interval <= 0)
      PRINT_INPUT_ERROR("REMD dump_xyz interval should be positive.");
    dump_prefix = dump_command[4];
    if (dump_prefix.empty() || dump_prefix.back() == '*')
      PRINT_INPUT_ERROR("REMD dump_xyz does not support an empty or '*' filename.");
    for (size_t i = 5; i < dump_command.size(); ++i) {
      if (dump_command[i] == "velocity")
        dump_velocity = true;
      else
        PRINT_INPUT_ERROR("REMD dump_xyz supports positions and optional velocity only.");
    }
    dump_xyz = true;
  }

  internal_seed = static_cast<unsigned int>(rand());
  if (internal_seed == 0)
    internal_seed = 1;
}

class REMD_Driver
{
public:
  REMD_Driver(
    const std::vector<std::vector<std::string>>& potential_commands,
    const std::vector<std::string>& ensemble_command,
    const std::vector<std::string>& multi_replica_command,
    const std::vector<std::string>& dump_xyz_command,
    Atom& source_atom,
    Box& source_box,
    std::vector<Group>& source_group,
    double time_step,
    int number_of_steps);

  ~REMD_Driver();
  void run();

private:
  REMD_Config config_;
  const std::vector<std::vector<std::string>>& potential_commands_;
  const std::vector<std::string>& ensemble_command_;
  Atom& source_atom_;
  Box& source_box_;
  std::vector<Group>& source_group_;
  double time_step_;
  int number_of_steps_;
  double temperature_coupling_ = 0.0;
  unsigned long long potential_hash_ = 0;
  long long completed_steps_ = 0;
  int exchange_count_ = 0;
  std::mt19937 exchange_rng_;
  std::vector<std::string> restart_bdp_states_;
  std::vector<int> replica_to_temperature_;
  std::vector<int> temperature_to_replica_;
  std::vector<int> exchange_attempts_;
  std::vector<int> exchange_accepts_;
  std::vector<double> exchange_probability_sum_;
  Replica_Runtime runtime_;
  std::vector<FILE*> trajectory_files_;

  void validate_input();
  void initialize_devices();
  void initialize_slot(Replica_Slot& slot);
  void set_temperature(Replica_Slot& slot, int temperature_label);
  void start_workers();
  void stop_workers();
  void advance_all(int steps);
  void equilibrate();
  void attempt_exchanges(long long global_step, FILE* exchange_file);
  void scale_velocity(Replica_Slot& slot, double factor);
  void write_temperature_ladder() const;
  void write_temperature_status(long long step, FILE* file) const;
  void write_thermo(long long step, FILE* file) const;
  void write_ladder_report() const;
  void open_trajectory_files();
  void close_trajectory_files();
  void write_trajectories(long long step);
  void write_restart();
  void write_restart_metadata() const;
  void write_restart_xyz(Replica_Slot& slot) const;
  void read_restart_metadata();
  void read_restart_xyz(Replica_Slot& slot);
  void validate_restart_mappings() const;
};

REMD_Driver::REMD_Driver(
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
  , runtime_(potential_commands, source_atom, source_box, source_group, time_step)
{
  config_.parse(multi_replica_command, dump_xyz_command);
}

REMD_Driver::~REMD_Driver()
{
  stop_workers();
  close_trajectory_files();
}

void REMD_Driver::validate_input()
{
  if (number_of_steps_ <= 0)
    PRINT_INPUT_ERROR("REMD run steps should be positive.");
  if (source_atom_.number_of_atoms <= 0)
    PRINT_INPUT_ERROR("REMD requires a non-empty model.");
  if (potential_commands_.size() != 1 || potential_commands_[0].size() != 2)
    PRINT_INPUT_ERROR("REMD currently requires exactly one standard NEP potential command.");
  const std::string potential_name = read_potential_name(potential_commands_[0][1]);
  if (!is_supported_replica_nep_name(potential_name))
    PRINT_INPUT_ERROR(
      "REMD explicit streams support energy NEP4/NEP5 and qNEP4 models.");
  if (is_supported_charge_nep_name(potential_name) &&
      (!source_box_.pbc_x || !source_box_.pbc_y || !source_box_.pbc_z))
    PRINT_INPUT_ERROR("qNEP REMD requires periodic boundaries in all directions.");
  potential_hash_ = hash_file(potential_commands_[0][1]);

  if (ensemble_command_.size() != 5 || ensemble_command_[0] != "ensemble" ||
      ensemble_command_[1] != "nvt_bdp")
    PRINT_INPUT_ERROR("REMD currently supports only ensemble nvt_bdp.");
  double input_temperature_1 = 0.0;
  double input_temperature_2 = 0.0;
  if (!is_valid_real(ensemble_command_[2].c_str(), &input_temperature_1) ||
      !is_valid_real(ensemble_command_[3].c_str(), &input_temperature_2) ||
      input_temperature_1 <= 0.0 || input_temperature_2 <= 0.0 ||
      !is_valid_real(ensemble_command_[4].c_str(), &temperature_coupling_) ||
      temperature_coupling_ < 1.0)
    PRINT_INPUT_ERROR("Invalid nvt_bdp parameters for REMD.");
  if (!values_match(input_temperature_1, config_.temperatures.front()) ||
      !values_match(input_temperature_2, config_.temperatures.back()))
    PRINT_INPUT_ERROR(
      "REMD ensemble temperatures should match the first and last ladder temperatures.");

  int number_of_devices = 0;
  CHECK(gpuGetDeviceCount(&number_of_devices));
  if (number_of_devices <= 0)
    PRINT_INPUT_ERROR("No GPU device is available for REMD.");
  if (config_.replicas_per_gpu == 0)
    config_.replicas_per_gpu =
      (config_.replicas + number_of_devices - 1) / number_of_devices;
  if (config_.replicas_per_gpu * number_of_devices < config_.replicas)
    PRINT_INPUT_ERROR(
      "REMD replicas_per_gpu is too small; batched replica swapping is not enabled.");

  replica_to_temperature_.resize(config_.replicas);
  temperature_to_replica_.resize(config_.replicas);
  exchange_attempts_.assign(config_.replicas - 1, 0);
  exchange_accepts_.assign(config_.replicas - 1, 0);
  exchange_probability_sum_.assign(config_.replicas - 1, 0.0);
  restart_bdp_states_.resize(config_.replicas);
  for (int r = 0; r < config_.replicas; ++r) {
    replica_to_temperature_[r] = r;
    temperature_to_replica_[r] = r;
  }

  if (config_.resume) {
    read_restart_metadata();
    validate_append_output_file("remd_exchange.out", exchange_output_marker);
    validate_append_output_file("remd_temperature_status.out", status_output_marker);
    validate_append_output_file("remd_thermo.out", thermo_output_marker);
  } else {
    exchange_rng_.seed(config_.internal_seed);
  }

  printf("Temperature replica-exchange molecular dynamics.\n");
  printf("    replicas: %d.\n", config_.replicas);
  printf("    replicas per GPU: %d.\n", config_.replicas_per_gpu);
  printf("    exchange interval: %d steps.\n", config_.exchange_interval);
  printf("    thermostat: nvt_bdp with tau_T = %g steps.\n", temperature_coupling_);
  printf(
    "    temperature range: %g K to %g K.\n",
    config_.temperatures.front(),
    config_.temperatures.back());
  if (config_.equilibration_steps > 0)
    printf("    independent equilibration: %d steps per replica.\n", config_.equilibration_steps);
  if (config_.resume)
    printf(
      "    restart prefix: %s at completed production step %lld.\n",
      config_.resume_prefix.c_str(),
      completed_steps_);
  if (config_.dump_xyz)
    printf(
      "    dump %d replica trajectories every %d steps%s.\n",
      config_.replicas,
      config_.dump_interval,
      config_.dump_velocity ? " with velocities" : "");
  fflush(stdout);
}

void REMD_Driver::set_temperature(Replica_Slot& slot, const int temperature_label)
{
  slot.temperature_label = temperature_label;
  slot.target_temperature = config_.temperatures[temperature_label];
  if (slot.thermostat)
    slot.thermostat->temperature = slot.target_temperature;
}

void REMD_Driver::initialize_slot(Replica_Slot& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  if (config_.resume)
    read_restart_xyz(slot);

  allocate_memory_gpu(slot.group, slot.atom, slot.thermo);
  if (config_.resume) {
    slot.atom.velocity_per_atom.copy_from_host(slot.atom.cpu_velocity_per_atom.data());
  } else {
    Velocity velocity;
    velocity.initialize(
      false,
      config_.temperatures[slot.replica_id],
      slot.atom,
      true,
      static_cast<int>(config_.internal_seed + 104729U * (slot.replica_id + 1U)));
  }

  double move_velocity[3] = {0.0, 0.0, 0.0};
  slot.thermostat.reset(new Ensemble_BDP(
    4,
    -1,
    move_velocity,
    config_.temperatures[replica_to_temperature_[slot.replica_id]],
    temperature_coupling_));
  set_temperature(slot, replica_to_temperature_[slot.replica_id]);
  if (config_.resume) {
    if (!slot.thermostat->import_rng_state(restart_bdp_states_[slot.replica_id]))
      PRINT_INPUT_ERROR("Invalid nvt_bdp RNG state in the REMD restart.");
  } else {
    slot.thermostat->seed_rng(
      config_.internal_seed + 130363U * (slot.replica_id + 1U) + 17U);
  }
}

void REMD_Driver::initialize_devices()
{
  runtime_.initialize(
    config_.replicas,
    config_.replicas_per_gpu,
    [this](Replica_Slot& slot) { initialize_slot(slot); });
}

void REMD_Driver::start_workers() { runtime_.start_workers(); }

void REMD_Driver::stop_workers() { runtime_.stop_workers(); }

void REMD_Driver::advance_all(const int steps) { runtime_.advance_all(steps); }

static __global__ void gpu_scale_replica_velocity(
  const int number_of_atoms,
  const double factor,
  double* velocity_x,
  double* velocity_y,
  double* velocity_z)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < number_of_atoms) {
    velocity_x[atom] *= factor;
    velocity_y[atom] *= factor;
    velocity_z[atom] *= factor;
  }
}

void REMD_Driver::scale_velocity(Replica_Slot& slot, const double factor)
{
  CHECK(gpuSetDevice(slot.device_id));
  const int number_of_atoms = slot.atom.number_of_atoms;
  gpu_scale_replica_velocity<<<
    (number_of_atoms - 1) / 128 + 1, 128, 0, slot.stream.get()>>>(
    number_of_atoms,
    factor,
    slot.atom.velocity_per_atom.data(),
    slot.atom.velocity_per_atom.data() + number_of_atoms,
    slot.atom.velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

void REMD_Driver::attempt_exchanges(const long long global_step, FILE* exchange_file)
{
  const int parity = exchange_count_ % 2;
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (int low_label = parity; low_label + 1 < config_.replicas; low_label += 2) {
    const int high_label = low_label + 1;
    const int low_replica = temperature_to_replica_[low_label];
    const int high_replica = temperature_to_replica_[high_label];
    Replica_Slot& low = *runtime_.slots()[low_replica];
    Replica_Slot& high = *runtime_.slots()[high_replica];
    const double beta_low = config_.betas[low_label];
    const double beta_high = config_.betas[high_label];
    const double exponent =
      (low.potential_energy - high.potential_energy) * (beta_low - beta_high);
    const double probability = exponent >= 0.0 ? 1.0 : std::exp(exponent);
    const bool accepted = exponent >= 0.0 || uniform(exchange_rng_) < probability;
    ++exchange_attempts_[low_label];
    exchange_probability_sum_[low_label] += probability;

    if (exchange_file) {
      fprintf(
        exchange_file,
        "%lld %d %d %d %d %.16g %.16g %.16g %.16g %.16g %.16g %d\n",
        global_step,
        low_replica,
        high_replica,
        low_label,
        high_label,
        config_.temperatures[low_label],
        config_.temperatures[high_label],
        low.potential_energy,
        high.potential_energy,
        exponent,
        probability,
        accepted ? 1 : 0);
    }
    if (!accepted)
      continue;

    ++exchange_accepts_[low_label];
    scale_velocity(
      low, std::sqrt(config_.temperatures[high_label] / config_.temperatures[low_label]));
    scale_velocity(
      high, std::sqrt(config_.temperatures[low_label] / config_.temperatures[high_label]));
    set_temperature(low, high_label);
    set_temperature(high, low_label);
    replica_to_temperature_[low_replica] = high_label;
    replica_to_temperature_[high_replica] = low_label;
    temperature_to_replica_[low_label] = high_replica;
    temperature_to_replica_[high_label] = low_replica;
  }
  ++exchange_count_;
  if (exchange_file)
    fflush(exchange_file);
}

void REMD_Driver::equilibrate()
{
  if (config_.equilibration_steps <= 0)
    return;
  printf(
    "    Equilibrating all replicas independently for %d steps.\n",
    config_.equilibration_steps);
  fflush(stdout);
  const int progress_interval = std::max(1, config_.equilibration_steps / 10);
  int completed = 0;
  while (completed < config_.equilibration_steps) {
    const int steps =
      std::min(progress_interval, config_.equilibration_steps - completed);
    advance_all(steps);
    completed += steps;
    printf("    %d REMD equilibration steps completed.\n", completed);
    fflush(stdout);
  }
}

void REMD_Driver::write_temperature_ladder() const
{
  FILE* file = my_fopen("remd_temps.in", "w");
  fprintf(file, "# REMD temperature ladder generated by GPUMD\n");
  fprintf(file, "# replicas %d\n", config_.replicas);
  for (const double temperature : config_.temperatures)
    fprintf(file, "%.16g\n", temperature);
  fclose(file);
}

void REMD_Driver::write_temperature_status(const long long step, FILE* file) const
{
  if (!file)
    return;
  fprintf(file, "%lld", step);
  for (const int temperature_label : replica_to_temperature_)
    fprintf(file, " %d", temperature_label);
  fprintf(file, "\n");
  fflush(file);
}

void REMD_Driver::write_thermo(const long long step, FILE* file) const
{
  if (!file)
    return;
  fprintf(file, "%lld", step);
  for (const Replica_Slot* slot : runtime_.slots()) {
    fprintf(
      file,
      " %.16g %.16g %.16g",
      slot->target_temperature,
      slot->potential_energy,
      slot->box.get_volume());
  }
  fprintf(file, "\n");
  fflush(file);
}

void REMD_Driver::write_ladder_report() const
{
  FILE* file = my_fopen("remd_ladder.out", "w");
  fprintf(file, "# REMD temperature ladder report\n");
  fprintf(file, "# target_acceptance %.10e\n", target_acceptance);
  fprintf(
    file,
    "# temp_i temp_j temperature_i temperature_j attempts accepted acceptance_ratio "
    "average_probability meets_target\n");
  bool all_meet_target = true;
  for (int label = 0; label + 1 < config_.replicas; ++label) {
    const int attempts = exchange_attempts_[label];
    const double ratio =
      attempts > 0 ? static_cast<double>(exchange_accepts_[label]) / attempts : 0.0;
    const double average =
      attempts > 0 ? exchange_probability_sum_[label] / attempts : 0.0;
    const bool meets_target = attempts > 0 && ratio >= target_acceptance;
    all_meet_target = all_meet_target && meets_target;
    fprintf(
      file,
      "%d %d %.16g %.16g %d %d %.10e %.10e %d\n",
      label,
      label + 1,
      config_.temperatures[label],
      config_.temperatures[label + 1],
      attempts,
      exchange_accepts_[label],
      ratio,
      average,
      meets_target ? 1 : 0);
  }
  fprintf(file, "# summary all_pairs_meet_target %d\n", all_meet_target ? 1 : 0);
  fclose(file);
}

std::string trajectory_filename(const std::string& prefix, const int replica)
{
  const std::string suffix = "_replica_" + std::to_string(replica);
  const size_t separator = prefix.find_last_of("/\\");
  const size_t extension = prefix.find_last_of('.');
  if (extension != std::string::npos &&
      (separator == std::string::npos || extension > separator))
    return prefix.substr(0, extension) + suffix + prefix.substr(extension);
  return prefix + suffix + ".xyz";
}

void REMD_Driver::open_trajectory_files()
{
  if (!config_.dump_xyz)
    return;
  trajectory_files_.resize(config_.replicas, nullptr);
  for (int replica = 0; replica < config_.replicas; ++replica)
    trajectory_files_[replica] = my_fopen(
      trajectory_filename(config_.dump_prefix, replica).c_str(),
      config_.resume ? "a" : "w");
}

void REMD_Driver::close_trajectory_files()
{
  for (FILE* file : trajectory_files_) {
    if (file)
      fclose(file);
  }
  trajectory_files_.clear();
}

void write_xyz_header(
  FILE* file,
  const Replica_Slot& slot,
  const long long step,
  const double time_step,
  const bool has_velocity)
{
  fprintf(file, "%d\n", slot.atom.number_of_atoms);
  fprintf(
    file,
    "Time=%.8f step=%lld replica=%d temperature_label=%d target_temperature=%.16g ",
    step * time_step * TIME_UNIT_CONVERSION,
    step,
    slot.replica_id,
    slot.temperature_label,
    slot.target_temperature);
  fprintf(
    file,
    "pbc=\"%c %c %c\" ",
    slot.box.pbc_x ? 'T' : 'F',
    slot.box.pbc_y ? 'T' : 'F',
    slot.box.pbc_z ? 'T' : 'F');
  fprintf(
    file,
    "Lattice=\"%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g\" ",
    slot.box.cpu_h[0],
    slot.box.cpu_h[3],
    slot.box.cpu_h[6],
    slot.box.cpu_h[1],
    slot.box.cpu_h[4],
    slot.box.cpu_h[7],
    slot.box.cpu_h[2],
    slot.box.cpu_h[5],
    slot.box.cpu_h[8]);
  fprintf(file, "Properties=species:S:1:pos:R:3%s\n", has_velocity ? ":vel:R:3" : "");
}

void REMD_Driver::write_trajectories(const long long step)
{
  if (!config_.dump_xyz)
    return;
  const int number_of_atoms = source_atom_.number_of_atoms;
  for (Replica_Slot* slot : runtime_.slots()) {
    CHECK(gpuSetDevice(slot->device_id));
    slot->stream.synchronize();
    slot->atom.position_per_atom.copy_to_host(slot->atom.cpu_position_per_atom.data());
    if (config_.dump_velocity)
      slot->atom.velocity_per_atom.copy_to_host(slot->atom.cpu_velocity_per_atom.data());
    FILE* file = trajectory_files_[slot->replica_id];
    write_xyz_header(file, *slot, step, time_step_, config_.dump_velocity);
    for (int atom = 0; atom < number_of_atoms; ++atom) {
      fprintf(
        file,
        "%s %.16g %.16g %.16g",
        source_atom_.cpu_atom_symbol[atom].c_str(),
        slot->atom.cpu_position_per_atom[atom],
        slot->atom.cpu_position_per_atom[atom + number_of_atoms],
        slot->atom.cpu_position_per_atom[atom + 2 * number_of_atoms]);
      if (config_.dump_velocity) {
        fprintf(
          file,
          " %.16g %.16g %.16g",
          slot->atom.cpu_velocity_per_atom[atom] / TIME_UNIT_CONVERSION,
          slot->atom.cpu_velocity_per_atom[atom + number_of_atoms] / TIME_UNIT_CONVERSION,
          slot->atom.cpu_velocity_per_atom[atom + 2 * number_of_atoms] /
            TIME_UNIT_CONVERSION);
      }
      fprintf(file, "\n");
    }
    fflush(file);
  }
}

void REMD_Driver::write_restart_xyz(Replica_Slot& slot) const
{
  CHECK(gpuSetDevice(slot.device_id));
  slot.stream.synchronize();
  slot.atom.position_per_atom.copy_to_host(slot.atom.cpu_position_per_atom.data());
  slot.atom.velocity_per_atom.copy_to_host(slot.atom.cpu_velocity_per_atom.data());

  const std::string filename =
    "remd_replica_" + std::to_string(slot.replica_id) + "_restart.xyz";
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
    "replica_mode=remd replica=%d temperature_label=%d target_temperature=%.17g "
    "potential_energy=%.17g restart_step=%lld\n",
    slot.replica_id,
    slot.temperature_label,
    slot.target_temperature,
    slot.potential_energy,
    completed_steps_ + number_of_steps_);
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
      slot.atom.cpu_velocity_per_atom[atom + number_of_atoms] / TIME_UNIT_CONVERSION,
      slot.atom.cpu_velocity_per_atom[atom + 2 * number_of_atoms] /
        TIME_UNIT_CONVERSION);
  }
  fclose(file);
}

void REMD_Driver::write_restart_metadata() const
{
  FILE* file = my_fopen("remd_restart.meta", "w");
  fprintf(file, "# GPUMD REMD restart metadata\n");
  fprintf(file, "format_version 3\n");
  fprintf(file, "mode remd\n");
  fprintf(file, "replicas %d\n", config_.replicas);
  fprintf(file, "completed_steps %lld\n", completed_steps_ + number_of_steps_);
  fprintf(file, "exchange_count %d\n", exchange_count_);
  fprintf(file, "exchange_interval %d\n", config_.exchange_interval);
  fprintf(file, "time_step %.17g\n", time_step_);
  fprintf(file, "temperature_coupling %.17g\n", temperature_coupling_);
  fprintf(file, "potential_hash %llu\n", potential_hash_);
  fprintf(file, "temperatures");
  for (const double temperature : config_.temperatures)
    fprintf(file, " %.17g", temperature);
  fprintf(file, "\nreplica_to_temperature");
  for (const int label : replica_to_temperature_)
    fprintf(file, " %d", label);
  fprintf(file, "\ntemperature_to_replica");
  for (const int replica : temperature_to_replica_)
    fprintf(file, " %d", replica);
  fprintf(file, "\nexchange_attempts");
  for (const int value : exchange_attempts_)
    fprintf(file, " %d", value);
  fprintf(file, "\nexchange_accepts");
  for (const int value : exchange_accepts_)
    fprintf(file, " %d", value);
  fprintf(file, "\nexchange_probability_sum");
  for (const double value : exchange_probability_sum_)
    fprintf(file, " %.17g", value);
  std::ostringstream exchange_state;
  exchange_state << exchange_rng_;
  fprintf(file, "\nexchange_rng %s\n", exchange_state.str().c_str());
  for (const Replica_Slot* slot : runtime_.slots()) {
    fprintf(
      file,
      "bdp_rng %d %s\n",
      slot->replica_id,
      slot->thermostat->export_rng_state().c_str());
  }
  fclose(file);
}

void REMD_Driver::write_restart()
{
  for (Replica_Slot* slot : runtime_.slots())
    write_restart_xyz(*slot);
  write_restart_metadata();
}

void REMD_Driver::validate_restart_mappings() const
{
  std::vector<bool> seen_temperature(config_.replicas, false);
  std::vector<bool> seen_replica(config_.replicas, false);
  for (int replica = 0; replica < config_.replicas; ++replica) {
    const int label = replica_to_temperature_[replica];
    if (label < 0 || label >= config_.replicas || seen_temperature[label])
      PRINT_INPUT_ERROR("REMD restart replica_to_temperature is not a permutation.");
    seen_temperature[label] = true;
  }
  for (int label = 0; label < config_.replicas; ++label) {
    const int replica = temperature_to_replica_[label];
    if (replica < 0 || replica >= config_.replicas || seen_replica[replica])
      PRINT_INPUT_ERROR("REMD restart temperature_to_replica is not a permutation.");
    seen_replica[replica] = true;
    if (replica_to_temperature_[replica] != label)
      PRINT_INPUT_ERROR("REMD restart temperature mappings are not inverse permutations.");
  }
}

void REMD_Driver::read_restart_metadata()
{
  const std::string filename = config_.resume_prefix + "_restart.meta";
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the REMD restart metadata file.");

  bool has_version = false;
  bool has_mode = false;
  bool has_replicas = false;
  bool has_completed_steps = false;
  bool has_exchange_count = false;
  bool has_exchange_interval = false;
  bool has_time_step = false;
  bool has_coupling = false;
  bool has_hash = false;
  bool has_temperatures = false;
  bool has_replica_mapping = false;
  bool has_temperature_mapping = false;
  bool has_attempts = false;
  bool has_accepts = false;
  bool has_probability = false;
  bool has_exchange_rng = false;
  std::vector<bool> has_bdp(config_.replicas, false);
  std::string key;
  while (input >> key) {
    if (!key.empty() && key[0] == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else if (key == "format_version") {
      int version = 0;
      if (has_version || !(input >> version) || version != 3)
        PRINT_INPUT_ERROR("Unsupported REMD restart format_version.");
      has_version = true;
    } else if (key == "mode") {
      std::string mode;
      if (has_mode || !(input >> mode) || mode != "remd")
        PRINT_INPUT_ERROR("REMD restart mode is invalid.");
      has_mode = true;
    } else if (key == "replicas") {
      int replicas = 0;
      if (has_replicas || !(input >> replicas) || replicas != config_.replicas)
        PRINT_INPUT_ERROR("REMD restart replicas do not match the input.");
      has_replicas = true;
    } else if (key == "completed_steps") {
      if (has_completed_steps || !(input >> completed_steps_) || completed_steps_ < 0)
        PRINT_INPUT_ERROR("REMD restart completed_steps is invalid.");
      has_completed_steps = true;
    } else if (key == "exchange_count") {
      if (has_exchange_count || !(input >> exchange_count_) || exchange_count_ < 0)
        PRINT_INPUT_ERROR("REMD restart exchange_count is invalid.");
      has_exchange_count = true;
    } else if (key == "exchange_interval") {
      int value = 0;
      if (has_exchange_interval || !(input >> value) ||
          value != config_.exchange_interval)
        PRINT_INPUT_ERROR("REMD restart exchange_interval does not match the input.");
      has_exchange_interval = true;
    } else if (key == "time_step") {
      double value = 0.0;
      if (has_time_step || !(input >> value) || !values_match(value, time_step_))
        PRINT_INPUT_ERROR("REMD restart time_step does not match the input.");
      has_time_step = true;
    } else if (key == "temperature_coupling") {
      double value = 0.0;
      if (has_coupling || !(input >> value) || !values_match(value, temperature_coupling_))
        PRINT_INPUT_ERROR("REMD restart nvt_bdp coupling does not match the input.");
      has_coupling = true;
    } else if (key == "potential_hash") {
      unsigned long long value = 0;
      if (has_hash || !(input >> value) || value != potential_hash_)
        PRINT_INPUT_ERROR("REMD restart potential file does not match the input.");
      has_hash = true;
    } else if (key == "temperatures") {
      if (has_temperatures)
        PRINT_INPUT_ERROR("REMD restart repeats temperatures.");
      for (int r = 0; r < config_.replicas; ++r) {
        double value = 0.0;
        if (!(input >> value) || !values_match(value, config_.temperatures[r]))
          PRINT_INPUT_ERROR("REMD restart temperature ladder does not match the input.");
      }
      has_temperatures = true;
    } else if (key == "replica_to_temperature") {
      if (has_replica_mapping)
        PRINT_INPUT_ERROR("REMD restart repeats replica_to_temperature.");
      for (int& value : replica_to_temperature_) {
        if (!(input >> value))
          PRINT_INPUT_ERROR("REMD restart replica_to_temperature is incomplete.");
      }
      has_replica_mapping = true;
    } else if (key == "temperature_to_replica") {
      if (has_temperature_mapping)
        PRINT_INPUT_ERROR("REMD restart repeats temperature_to_replica.");
      for (int& value : temperature_to_replica_) {
        if (!(input >> value))
          PRINT_INPUT_ERROR("REMD restart temperature_to_replica is incomplete.");
      }
      has_temperature_mapping = true;
    } else if (key == "exchange_attempts") {
      if (has_attempts)
        PRINT_INPUT_ERROR("REMD restart repeats exchange_attempts.");
      for (int& value : exchange_attempts_) {
        if (!(input >> value) || value < 0)
          PRINT_INPUT_ERROR("REMD restart exchange_attempts is invalid.");
      }
      has_attempts = true;
    } else if (key == "exchange_accepts") {
      if (has_accepts)
        PRINT_INPUT_ERROR("REMD restart repeats exchange_accepts.");
      for (int& value : exchange_accepts_) {
        if (!(input >> value) || value < 0)
          PRINT_INPUT_ERROR("REMD restart exchange_accepts is invalid.");
      }
      has_accepts = true;
    } else if (key == "exchange_probability_sum") {
      if (has_probability)
        PRINT_INPUT_ERROR("REMD restart repeats exchange_probability_sum.");
      for (double& value : exchange_probability_sum_) {
        if (!(input >> value) || !std::isfinite(value) || value < 0.0)
          PRINT_INPUT_ERROR("REMD restart exchange_probability_sum is invalid.");
      }
      has_probability = true;
    } else if (key == "exchange_rng") {
      if (has_exchange_rng)
        PRINT_INPUT_ERROR("REMD restart repeats exchange_rng.");
      std::string state;
      std::getline(input, state);
      std::istringstream state_input(trim_left(state));
      std::string trailing;
      if (!(state_input >> exchange_rng_) || state_input >> trailing)
        PRINT_INPUT_ERROR("REMD restart exchange_rng is invalid.");
      has_exchange_rng = true;
    } else if (key == "bdp_rng") {
      int replica = -1;
      if (!(input >> replica) || replica < 0 || replica >= config_.replicas || has_bdp[replica])
        PRINT_INPUT_ERROR("REMD restart bdp_rng replica id is invalid.");
      std::getline(input, restart_bdp_states_[replica]);
      restart_bdp_states_[replica] = trim_left(restart_bdp_states_[replica]);
      if (restart_bdp_states_[replica].empty())
        PRINT_INPUT_ERROR("REMD restart bdp_rng state is empty.");
      has_bdp[replica] = true;
    } else {
      PRINT_INPUT_ERROR("Unknown key in the REMD restart metadata.");
    }
  }

  if (!has_version || !has_mode || !has_replicas || !has_completed_steps ||
      !has_exchange_count || !has_exchange_interval || !has_time_step ||
      !has_coupling || !has_hash || !has_temperatures || !has_replica_mapping ||
      !has_temperature_mapping || !has_attempts || !has_accepts ||
      !has_probability || !has_exchange_rng ||
      std::find(has_bdp.begin(), has_bdp.end(), false) != has_bdp.end())
    PRINT_INPUT_ERROR("REMD restart metadata is incomplete.");
  if (exchange_count_ != completed_steps_ / config_.exchange_interval)
    PRINT_INPUT_ERROR("REMD restart exchange_count is inconsistent with completed_steps.");
  for (size_t i = 0; i < exchange_attempts_.size(); ++i) {
    const int expected_attempts =
      i % 2 == 0 ? exchange_count_ / 2 + exchange_count_ % 2 : exchange_count_ / 2;
    if (exchange_attempts_[i] != expected_attempts ||
        exchange_accepts_[i] > exchange_attempts_[i] ||
        exchange_probability_sum_[i] > exchange_attempts_[i] + 1.0e-12)
      PRINT_INPUT_ERROR("REMD restart exchange statistics are inconsistent.");
  }
  validate_restart_mappings();
}

void REMD_Driver::read_restart_xyz(Replica_Slot& slot)
{
  const std::string filename = config_.resume_prefix + "_replica_" +
                               std::to_string(slot.replica_id) + "_restart.xyz";
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open an REMD replica restart xyz file.");
  int number_of_atoms = 0;
  if (!(input >> number_of_atoms) || number_of_atoms != source_atom_.number_of_atoms)
    PRINT_INPUT_ERROR("REMD restart atom count does not match model.xyz.");
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string comment;
  if (!std::getline(input, comment))
    PRINT_INPUT_ERROR("Cannot read the REMD restart xyz comment line.");
  validate_xyz_box(comment, source_box_, "REMD");

  std::string value;
  int file_replica = -1;
  int temperature_label = -1;
  long long restart_step = -1;
  double target_temperature = 0.0;
  double potential_energy = 0.0;
  if (!extract_token_value(comment, "replica_mode", value) || value != "remd" ||
      !extract_token_value(comment, "Properties", value) ||
      value != "species:S:1:pos:R:3:mass:R:1:vel:R:3" ||
      !extract_token_value(comment, "replica", value) ||
      !parse_scalar(value, file_replica) || file_replica != slot.replica_id ||
      !extract_token_value(comment, "temperature_label", value) ||
      !parse_scalar(value, temperature_label) ||
      temperature_label != replica_to_temperature_[slot.replica_id] ||
      !extract_token_value(comment, "target_temperature", value) ||
      !parse_scalar(value, target_temperature) ||
      !values_match(target_temperature, config_.temperatures[temperature_label]) ||
      !extract_token_value(comment, "potential_energy", value) ||
      !parse_scalar(value, potential_energy) || !std::isfinite(potential_energy) ||
      !extract_token_value(comment, "restart_step", value) ||
      !parse_scalar(value, restart_step) || restart_step != completed_steps_)
    PRINT_INPUT_ERROR("REMD restart xyz metadata does not match the restart mapping.");
  slot.temperature_label = temperature_label;
  slot.target_temperature = target_temperature;
  slot.potential_energy = potential_energy;

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
      PRINT_INPUT_ERROR("REMD restart xyz atom data is incomplete.");
    if (symbol != source_atom_.cpu_atom_symbol[atom] ||
        !values_match(mass, source_atom_.cpu_mass[atom]))
      PRINT_INPUT_ERROR("REMD restart xyz species or mass does not match model.xyz.");
    for (int d = 0; d < 3; ++d) {
      const int index = atom + d * n;
      if (!std::isfinite(slot.atom.cpu_position_per_atom[index]) ||
          !std::isfinite(slot.atom.cpu_velocity_per_atom[index]))
        PRINT_INPUT_ERROR("REMD restart xyz contains a non-finite atom value.");
      slot.atom.cpu_velocity_per_atom[index] *= TIME_UNIT_CONVERSION;
    }
  }
}

void REMD_Driver::run()
{
  validate_input();
  const auto initialization_begin = std::chrono::high_resolution_clock::now();
  initialize_devices();
  start_workers();
  if (!config_.resume)
    equilibrate();
  const auto initialization_end = std::chrono::high_resolution_clock::now();

  write_temperature_ladder();
  const bool append_exchange = config_.resume && file_has_content("remd_exchange.out");
  const bool append_status =
    config_.resume && file_has_content("remd_temperature_status.out");
  const bool append_thermo = config_.resume && file_has_content("remd_thermo.out");
  FILE* exchange_file =
    my_fopen("remd_exchange.out", config_.resume ? "a" : "w");
  FILE* status_file =
    my_fopen("remd_temperature_status.out", config_.resume ? "a" : "w");
  FILE* thermo_file = my_fopen("remd_thermo.out", config_.resume ? "a" : "w");
  if (!append_exchange) {
    fprintf(exchange_file, "%s\n", exchange_output_marker);
    fprintf(exchange_file, "# Exchange states are recorded immediately before swapping.\n");
    fprintf(exchange_file, "# Units: target_temperature K, potential_energy eV.\n");
    fprintf(
      exchange_file,
      "# step replica_i replica_j temperature_label_i temperature_label_j "
      "target_temperature_i target_temperature_j potential_energy_i "
      "potential_energy_j log_acceptance_ratio acceptance_probability accepted\n");
  }
  if (!append_status) {
    fprintf(status_file, "%s\n", status_output_marker);
    fprintf(
      status_file,
      "# Replica-to-temperature mapping after any exchange at the reported step.\n");
    fprintf(status_file, "# step");
    for (int replica = 0; replica < config_.replicas; ++replica)
      fprintf(status_file, " temperature_label_of_replica_%d", replica);
    fprintf(status_file, "\n");
  }
  if (!append_thermo) {
    fprintf(thermo_file, "%s\n", thermo_output_marker);
    fprintf(thermo_file, "# Replica states after any exchange at the reported step.\n");
    fprintf(thermo_file, "# Units: target_temperature K, potential_energy eV, volume A^3.\n");
    fprintf(thermo_file, "# step");
    for (int replica = 0; replica < config_.replicas; ++replica) {
      fprintf(
        thermo_file,
        " target_temperature_%d potential_energy_%d volume_%d",
        replica,
        replica,
        replica);
    }
    fprintf(thermo_file, "\n");
  }
  open_trajectory_files();
  if (!append_status)
    write_temperature_status(completed_steps_, status_file);
  if (!append_thermo)
    write_thermo(completed_steps_, thermo_file);

  const auto production_begin = std::chrono::high_resolution_clock::now();
  const int progress_interval = std::max(1, number_of_steps_ / 10);
  int local_step = 0;
  while (local_step < number_of_steps_) {
    const long long global_step = completed_steps_ + local_step;
    int next_event = next_periodic_event(
      local_step, number_of_steps_, global_step, config_.exchange_interval);
    if (config_.dump_xyz) {
      next_event = std::min(
        next_event,
        next_periodic_event(
          local_step, number_of_steps_, global_step, config_.dump_interval));
    }
    const int next_progress =
      std::min(number_of_steps_, ((local_step / progress_interval) + 1) * progress_interval);
    next_event = std::min(next_event, next_progress);
    advance_all(next_event - local_step);
    local_step = next_event;
    const long long current_global_step = completed_steps_ + local_step;

    const bool exchange_due = current_global_step % config_.exchange_interval == 0;
    const bool dump_due =
      config_.dump_xyz && current_global_step % config_.dump_interval == 0;
    if (exchange_due)
      attempt_exchanges(current_global_step, exchange_file);
    write_thermo(current_global_step, thermo_file);
    if (exchange_due || dump_due || local_step == number_of_steps_)
      write_temperature_status(current_global_step, status_file);
    if (dump_due)
      write_trajectories(current_global_step);
    if (local_step % progress_interval == 0 || local_step == number_of_steps_) {
      printf("    %d REMD production steps completed.\n", local_step);
      fflush(stdout);
    }
  }
  const auto production_end = std::chrono::high_resolution_clock::now();

  stop_workers();
  write_restart();
  write_ladder_report();
  close_trajectory_files();
  fclose(exchange_file);
  fclose(status_file);
  fclose(thermo_file);

  const std::chrono::duration<double> initialization_time =
    initialization_end - initialization_begin;
  const std::chrono::duration<double> production_time = production_end - production_begin;
  print_line_1();
  printf("REMD initialization/equilibration time = %g second.\n", initialization_time.count());
  printf("Time used for this REMD production run = %g second.\n", production_time.count());
  printf(
    "Speed of this REMD run = %g atom*step/second.\n",
    source_atom_.number_of_atoms *
      (static_cast<double>(number_of_steps_) * config_.replicas / production_time.count()));
  print_line_2();
}

} // namespace

void replica::run_remd(
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
  REMD_Driver driver(
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
