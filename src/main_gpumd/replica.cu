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

#include "replica.cuh"
#include "integrate/ensemble_bdp.cuh"
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
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <thread>

static std::vector<int> make_logical_ids(int count)
{
  std::vector<int> ids(count);
  for (int i = 0; i < count; ++i) {
    ids[i] = i;
  }
  return ids;
}

struct ScopedTimer
{
  explicit ScopedTimer(double& target)
    : target_(target), start_(std::chrono::high_resolution_clock::now())
  {
  }

  ~ScopedTimer()
  {
    const auto finish = std::chrono::high_resolution_clock::now();
    target_ += std::chrono::duration<double>(finish - start_).count();
  }

  double& target_;
  std::chrono::high_resolution_clock::time_point start_;
};

static void read_temperatures_from_file(const char* filename, std::vector<double>& temperatures)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open REMD temperature file.");
  }

  while (input.peek() != EOF) {
    const std::vector<std::string> tokens = get_tokens(input);
    for (const auto& token : tokens) {
      if (token.empty() || token[0] == '#') {
        break;
      }
      double temperature = 0.0;
      if (!is_valid_real(token.c_str(), &temperature) || temperature <= 0.0) {
        PRINT_INPUT_ERROR("REMD temperatures must be positive real numbers.");
      }
      temperatures.push_back(temperature);
    }
  }
}

static void generate_temperatures_from_range(MultiReplicaConfig& config)
{
  config.temperatures.resize(config.replicas);
  if (config.temperature_style == MultiReplicaConfig::linear) {
    const double delta =
      (config.temperature_max - config.temperature_min) / (config.replicas - 1);
    for (int r = 0; r < config.replicas; ++r) {
      config.temperatures[r] = config.temperature_min + delta * r;
    }
  } else {
    const double ratio = config.temperature_max / config.temperature_min;
    for (int r = 0; r < config.replicas; ++r) {
      config.temperatures[r] =
        config.temperature_min * std::pow(ratio, double(r) / double(config.replicas - 1));
    }
  }
  config.generated_temperatures = true;
}

void MultiReplicaConfig::parse(const char** param, int num_param)
{
  if (num_param < 3) {
    PRINT_INPUT_ERROR("multi_replica should specify remd and its options.");
  }
  if (enabled()) {
    PRINT_INPUT_ERROR("Only one multi_replica command is allowed before run.");
  }
  if (strcmp(param[1], "remd") != 0) {
    PRINT_INPUT_ERROR("multi_replica currently supports only remd.");
  }
  mode = remd;

  enum TemperatureSource { unset, file, range };
  TemperatureSource temperature_source = unset;
  int i = 2;
  while (i < num_param) {
    if (strcmp(param[i], "replicas") == 0) {
      if (++i >= num_param || !is_valid_int(param[i], &replicas) || replicas <= 1) {
        PRINT_INPUT_ERROR("replicas should be an integer larger than 1.");
      }
      ++i;
      if (i < num_param && is_valid_int(param[i], &replicas_per_gpu)) {
        if (replicas_per_gpu <= 0) {
          PRINT_INPUT_ERROR("replicas_per_gpu should be positive.");
        }
        ++i;
      }
    } else if (strcmp(param[i], "exchange") == 0) {
      if (++i >= num_param || !is_valid_int(param[i], &exchange_interval) ||
          exchange_interval <= 0) {
        PRINT_INPUT_ERROR("exchange should be a positive integer.");
      }
      ++i;
    } else if (
      strcmp(param[i], "equilibrate") == 0 || strcmp(param[i], "equilibration") == 0 ||
      strcmp(param[i], "equilibration_steps") == 0) {
      if (++i >= num_param || !is_valid_int(param[i], &equilibration_steps) ||
          equilibration_steps < 0) {
        PRINT_INPUT_ERROR("equilibration should be a non-negative integer.");
      }
      ++i;
    } else if (strcmp(param[i], "temp") == 0) {
      if (temperature_source != unset) {
        PRINT_INPUT_ERROR("Specify only one temp source.");
      }
      if (++i >= num_param) {
        PRINT_INPUT_ERROR("temp requires a file name or Tmin Tmax.");
      }
      if (is_valid_real(param[i], &temperature_min)) {
        temperature_source = range;
        has_temperature_range = true;
        if (++i >= num_param || !is_valid_real(param[i], &temperature_max) ||
            temperature_min <= 0.0 || temperature_max <= temperature_min) {
          PRINT_INPUT_ERROR("temp should satisfy 0 < Tmin < Tmax.");
        }
        ++i;
      } else {
        temperature_source = file;
        read_temperatures_from_file(param[i++], temperatures);
      }
    } else if (strcmp(param[i], "spacing") == 0) {
      if (++i >= num_param) {
        PRINT_INPUT_ERROR("spacing requires geometric or linear.");
      }
      if (strcmp(param[i], "geometric") == 0) {
        temperature_style = geometric;
      } else if (strcmp(param[i], "linear") == 0) {
        temperature_style = linear;
      } else {
        PRINT_INPUT_ERROR("spacing should be geometric or linear.");
      }
      ++i;
    } else if (strcmp(param[i], "verbose") == 0 || strcmp(param[i], "verbose_output") == 0) {
      verbose_output = true;
      ++i;
    } else if (strcmp(param[i], "resume") == 0) {
      if (++i >= num_param) {
        PRINT_INPUT_ERROR("resume requires a restart prefix, for example: resume remd.");
      }
      resume = true;
      resume_prefix = param[i++];
    } else if (strcmp(param[i], "dump_xyz") == 0) {
      PRINT_INPUT_ERROR(
        "Do not put dump_xyz inside multi_replica. Use the ordinary GPUMD command, "
        "for example: dump_xyz -1 0 10000 remd_traj");
    } else {
      PRINT_INPUT_ERROR("Unknown multi_replica option.");
    }
  }

  if (replicas <= 1) {
    PRINT_INPUT_ERROR("REMD requires replicas larger than 1.");
  }
  if (exchange_interval <= 0) {
    PRINT_INPUT_ERROR("REMD requires exchange.");
  }
  if (has_temperature_range && temperatures.empty()) {
    generate_temperatures_from_range(*this);
  }
  if (static_cast<int>(temperatures.size()) != replicas) {
    PRINT_INPUT_ERROR("REMD requires exactly one temperature for each replica.");
  }
  for (int r = 1; r < replicas; ++r) {
    if (temperatures[r] <= temperatures[r - 1]) {
      PRINT_INPUT_ERROR("REMD temperatures should be strictly increasing.");
    }
  }

  int num_devices = 0;
  CHECK(gpuGetDeviceCount(&num_devices));
  if (num_devices <= 0) {
    PRINT_INPUT_ERROR("No GPU device is available.");
  }
  devices.clear();
  for (int d = 0; d < num_devices; ++d) {
    devices.push_back(d);
  }
  if (replicas_per_gpu == 0) {
    replicas_per_gpu = (replicas + num_devices - 1) / num_devices;
  }

  seed = rand();
  if (seed <= 0) {
    seed = 1;
  }
  betas.resize(replicas);
  for (int r = 0; r < replicas; ++r) {
    betas[r] = 1.0 / (K_B * temperatures[r]);
  }

  printf("Multi-replica REMD with %d replicas.\n", replicas);
  printf("    exchange interval is %d steps.\n", exchange_interval);
  if (generated_temperatures) {
    printf(
      "    generated %d temperatures from %g K to %g K using %s spacing.\n",
      replicas,
      temperature_min,
      temperature_max,
      temperature_style == geometric ? "geometric" : "linear");
  }
  if (equilibration_steps > 0) {
    printf(
      "    equilibrate each replica for %d steps before production exchanges.\n",
      equilibration_steps);
  }
  if (verbose_output) {
    printf("    verbose REMD logs are enabled.\n");
  }
  if (resume) {
    if (equilibration_steps > 0) {
      PRINT_INPUT_ERROR("REMD resume cannot be combined with equilibration.");
    }
    printf("    REMD restart will be read from prefix '%s'.\n", resume_prefix.c_str());
  }
  printf("    REMD temperature ladder will be written to remd_temps.in.\n");
  printf("    replicas per GPU is %d.\n", replicas_per_gpu);
  printf("    detected GPU devices:");
  for (int device : devices) {
    printf(" %d", device);
  }
  printf("\n");
  printf("    replicas will be assigned automatically across visible GPUs.\n");
}

void MultiReplicaConfig::parse_dump_xyz(const char** param, int num_param)
{
  if (mode != remd) {
    PRINT_INPUT_ERROR("dump_xyz must follow multi_replica remd.");
  }
  if (dump_xyz) {
    PRINT_INPUT_ERROR("REMD currently supports only one dump_xyz command.");
  }
  if (num_param < 5) {
    PRINT_INPUT_ERROR("dump_xyz should have at least 4 parameters.");
  }

  int grouping_method = -1;
  if (!is_valid_int(param[1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method of dump_xyz should be integer.");
  }
  if (grouping_method >= 0) {
    PRINT_INPUT_ERROR("REMD dump_xyz writes whole-system replica trajectories; use grouping_method -1.");
  }

  int group_id = -1;
  if (!is_valid_int(param[2], &group_id)) {
    PRINT_INPUT_ERROR("group id of dump_xyz should be integer.");
  }
  if (!is_valid_int(param[3], &dump_xyz_interval) || dump_xyz_interval <= 0) {
    PRINT_INPUT_ERROR("dump_xyz interval should be a positive integer.");
  }

  dump_xyz_filename = param[4];
  if (dump_xyz_filename.empty()) {
    PRINT_INPUT_ERROR("dump_xyz filename should not be empty.");
  }
  if (dump_xyz_filename.back() == '*') {
    PRINT_INPUT_ERROR("REMD dump_xyz does not support separated '*' filenames.");
  }

  dump_xyz_has_velocity = 0;
  for (int m = 5; m < num_param; ++m) {
    if (strcmp(param[m], "velocity") == 0) {
      dump_xyz_has_velocity = 1;
    } else {
      PRINT_INPUT_ERROR("REMD dump_xyz currently supports positions and optional velocity only.");
    }
  }

  dump_xyz = true;
  printf("REMD dump_xyz.\n");
  printf("    output one trajectory for each replica.\n");
  printf("    every %d steps.\n", dump_xyz_interval);
  printf(
    "    using file prefix %s%s.\n",
    dump_xyz_filename.c_str(),
    dump_xyz_has_velocity ? " with velocities" : "");
}

static std::vector<const char*> make_param_array(const std::vector<std::string>& tokens)
{
  std::vector<const char*> param(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    param[i] = tokens[i].c_str();
  }
  return param;
}

ReplicaContext::~ReplicaContext()
{
  if (device_id >= 0) {
    CHECK(gpuSetDevice(device_id));
  }
}

MultiReplicaDriver::MultiReplicaDriver(
  MultiReplicaConfig& multi_replica,
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  const std::vector<std::vector<std::string>>& potential_commands,
  const std::vector<std::string>& ensemble_command,
  double time_step,
  int number_of_steps)
  : config_(multi_replica)
  , source_atom_(atom)
  , source_box_(box)
  , source_group_(group)
  , potential_commands_(potential_commands)
  , ensemble_command_(ensemble_command)
  , time_step_(time_step)
  , number_of_steps_(number_of_steps)
{
}

void MultiReplicaDriver::run()
{
  const auto total_begin = std::chrono::high_resolution_clock::now();
  {
    ScopedTimer timer(timing_.initialize);
    initialize_replicas();
  }
  run_remd();
  const auto total_finish = std::chrono::high_resolution_clock::now();
  timing_.total = std::chrono::duration<double>(total_finish - total_begin).count();
  print_timing_summary("REMD");
}

void MultiReplicaDriver::initialize_replicas()
{
  if (potential_commands_.empty()) {
    PRINT_INPUT_ERROR("multi_replica requires at least one potential command.");
  }
  if (ensemble_command_.empty()) {
    PRINT_INPUT_ERROR("multi_replica requires an ensemble command before run.");
  }
  int num_devices = 0;
  CHECK(gpuGetDeviceCount(&num_devices));
  for (int device : config_.devices) {
    if (device < 0 || device >= num_devices) {
      PRINT_INPUT_ERROR("multi_replica device index is out of range.");
    }
  }

  initialize_logical_replicas();
  initialize_gpu_slots();
}

void MultiReplicaDriver::initialize_logical_replicas()
{
  replica_to_temperature_.resize(config_.replicas);
  temperature_to_replica_.resize(config_.replicas);
  logical_replicas_.resize(config_.replicas);
  exchange_attempts_.assign(std::max(0, config_.replicas - 1), 0);
  exchange_accepts_.assign(std::max(0, config_.replicas - 1), 0);
  exchange_probability_sum_.assign(std::max(0, config_.replicas - 1), 0.0);

  for (int r = 0; r < config_.replicas; ++r) {
    logical_replicas_[r].replica_id = r;
    logical_replicas_[r].temperature_label = r;
    logical_replicas_[r].target_temperature = config_.temperatures[r];
    logical_replicas_[r].box = source_box_;
    logical_replicas_[r].position_per_atom = source_atom_.cpu_position_per_atom;
    logical_replicas_[r].velocity_per_atom = source_atom_.cpu_velocity_per_atom;
    replica_to_temperature_[r] = r;
    temperature_to_replica_[r] = r;
  }

  if (config_.resume) {
    read_remd_restart();
  } else {
    for (int r = 0; r < config_.replicas; ++r) {
      randomize_logical_replica_velocity(
        logical_replicas_[r], config_.seed + 104729 * (r + 1));
    }
    printf("    REMD initial velocities are sampled from each replica target temperature.\n");
    fflush(stdout);
  }
}

static bool extract_quoted_value(const std::string& line, const std::string& key, std::string& value)
{
  const std::string needle = key + "=\"";
  const size_t start = line.find(needle);
  if (start == std::string::npos) {
    return false;
  }
  const size_t value_start = start + needle.size();
  const size_t value_end = line.find('"', value_start);
  if (value_end == std::string::npos) {
    return false;
  }
  value = line.substr(value_start, value_end - value_start);
  return true;
}

static bool extract_token_value(const std::string& line, const std::string& key, std::string& value)
{
  const std::string needle = key + "=";
  const size_t start = line.find(needle);
  if (start == std::string::npos) {
    return false;
  }
  const size_t value_start = start + needle.size();
  const size_t value_end = line.find_first_of(" \t\r\n", value_start);
  value = line.substr(value_start, value_end == std::string::npos ? std::string::npos : value_end - value_start);
  return true;
}

static bool parse_int_value(const std::string& text, int& value)
{
  std::istringstream input(text);
  std::string trailing;
  return (input >> value) && !(input >> trailing);
}

static bool parse_double_value(const std::string& text, double& value)
{
  std::istringstream input(text);
  std::string trailing;
  return (input >> value) && !(input >> trailing) && std::isfinite(value);
}

static bool values_match(double first, double second)
{
  const double scale = std::max(1.0, std::max(std::fabs(first), std::fabs(second)));
  return std::fabs(first - second) <= 1.0e-10 * scale;
}

void MultiReplicaDriver::read_remd_restart()
{
  read_remd_restart_metadata(config_.resume_prefix + "_restart.meta");
  for (int r = 0; r < config_.replicas; ++r) {
    read_remd_restart_xyz(
      config_.resume_prefix + "_replica_" + std::to_string(r) + "_restart.xyz",
      r,
      logical_replicas_[r]);
  }

  printf(
    "    REMD restart loaded from prefix '%s' (ensemble-internal state is reinitialized).\n",
    config_.resume_prefix.c_str());
}

void MultiReplicaDriver::read_remd_restart_metadata(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open REMD restart metadata file.");
  }

  bool has_format_version = false;
  bool has_mode = false;
  bool has_replicas = false;
  bool has_exchange_count = false;
  bool has_replica_to_temperature = false;
  bool has_temperature_to_replica = false;
  bool has_exchange_attempts = false;
  bool has_exchange_accepts = false;
  bool has_exchange_probability_sum = false;

  std::string key;
  while (input >> key) {
    if (key[0] == '#') {
      input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else if (key == "format_version") {
      int format_version = 0;
      if (has_format_version || !(input >> format_version) || format_version != 1) {
        PRINT_INPUT_ERROR("REMD restart metadata has an unsupported format_version.");
      }
      has_format_version = true;
    } else if (key == "mode") {
      std::string mode;
      if (has_mode || !(input >> mode) || mode != "remd") {
        PRINT_INPUT_ERROR("REMD restart metadata is not for REMD.");
      }
      has_mode = true;
    } else if (key == "replicas") {
      int replicas = 0;
      if (has_replicas || !(input >> replicas) || replicas != config_.replicas) {
        PRINT_INPUT_ERROR("REMD restart replicas do not match input replicas.");
      }
      has_replicas = true;
    } else if (key == "exchange_count") {
      if (has_exchange_count || !(input >> remd_exchange_count_)) {
        PRINT_INPUT_ERROR("REMD restart exchange_count is invalid.");
      }
      has_exchange_count = true;
    } else if (key == "replica_to_temperature") {
      if (has_replica_to_temperature) {
        PRINT_INPUT_ERROR("REMD restart metadata repeats replica_to_temperature.");
      }
      for (int r = 0; r < config_.replicas; ++r) {
        if (!(input >> replica_to_temperature_[r])) {
          PRINT_INPUT_ERROR("REMD restart replica_to_temperature is incomplete.");
        }
      }
      has_replica_to_temperature = true;
    } else if (key == "temperature_to_replica") {
      if (has_temperature_to_replica) {
        PRINT_INPUT_ERROR("REMD restart metadata repeats temperature_to_replica.");
      }
      for (int r = 0; r < config_.replicas; ++r) {
        if (!(input >> temperature_to_replica_[r])) {
          PRINT_INPUT_ERROR("REMD restart temperature_to_replica is incomplete.");
        }
      }
      has_temperature_to_replica = true;
    } else if (key == "exchange_attempts") {
      if (has_exchange_attempts) {
        PRINT_INPUT_ERROR("REMD restart metadata repeats exchange_attempts.");
      }
      for (int i = 0; i + 1 < config_.replicas; ++i) {
        if (!(input >> exchange_attempts_[i])) {
          PRINT_INPUT_ERROR("REMD restart exchange_attempts is incomplete.");
        }
      }
      has_exchange_attempts = true;
    } else if (key == "exchange_accepts") {
      if (has_exchange_accepts) {
        PRINT_INPUT_ERROR("REMD restart metadata repeats exchange_accepts.");
      }
      for (int i = 0; i + 1 < config_.replicas; ++i) {
        if (!(input >> exchange_accepts_[i])) {
          PRINT_INPUT_ERROR("REMD restart exchange_accepts is incomplete.");
        }
      }
      has_exchange_accepts = true;
    } else if (key == "exchange_probability_sum") {
      if (has_exchange_probability_sum) {
        PRINT_INPUT_ERROR("REMD restart metadata repeats exchange_probability_sum.");
      }
      for (int i = 0; i + 1 < config_.replicas; ++i) {
        if (!(input >> exchange_probability_sum_[i])) {
          PRINT_INPUT_ERROR("REMD restart exchange_probability_sum is incomplete.");
        }
      }
      has_exchange_probability_sum = true;
    } else {
      PRINT_INPUT_ERROR("Unknown key in REMD restart metadata.");
    }
  }

  if (
    !has_format_version || !has_mode || !has_replicas || !has_exchange_count ||
    !has_replica_to_temperature || !has_temperature_to_replica || !has_exchange_attempts ||
    !has_exchange_accepts || !has_exchange_probability_sum) {
    PRINT_INPUT_ERROR("REMD restart metadata is incomplete.");
  }
  validate_remd_restart_metadata();
}

void MultiReplicaDriver::validate_remd_restart_metadata() const
{
  if (remd_exchange_count_ < 0) {
    PRINT_INPUT_ERROR("REMD restart exchange_count should be non-negative.");
  }

  std::vector<bool> seen_temperature(config_.replicas, false);
  for (int replica_id = 0; replica_id < config_.replicas; ++replica_id) {
    const int temperature_label = replica_to_temperature_[replica_id];
    if (temperature_label < 0 || temperature_label >= config_.replicas ||
        seen_temperature[temperature_label]) {
      PRINT_INPUT_ERROR("REMD restart replica_to_temperature is not a permutation.");
    }
    seen_temperature[temperature_label] = true;
  }

  std::vector<bool> seen_replica(config_.replicas, false);
  for (int temperature_label = 0; temperature_label < config_.replicas; ++temperature_label) {
    const int replica_id = temperature_to_replica_[temperature_label];
    if (replica_id < 0 || replica_id >= config_.replicas || seen_replica[replica_id]) {
      PRINT_INPUT_ERROR("REMD restart temperature_to_replica is not a permutation.");
    }
    seen_replica[replica_id] = true;
  }

  for (int replica_id = 0; replica_id < config_.replicas; ++replica_id) {
    const int temperature_label = replica_to_temperature_[replica_id];
    if (temperature_to_replica_[temperature_label] != replica_id) {
      PRINT_INPUT_ERROR("REMD restart temperature mappings are not inverse permutations.");
    }
  }

  for (int i = 0; i + 1 < config_.replicas; ++i) {
    if (exchange_attempts_[i] < 0 || exchange_accepts_[i] < 0 ||
        exchange_accepts_[i] > exchange_attempts_[i] ||
        !std::isfinite(exchange_probability_sum_[i]) || exchange_probability_sum_[i] < 0.0) {
      PRINT_INPUT_ERROR("REMD restart exchange statistics are invalid.");
    }
  }
}

void MultiReplicaDriver::read_remd_restart_xyz(
  const std::string& filename, int replica_id, LogicalReplicaState& logical_replica)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Cannot open REMD replica restart xyz file.");
  }

  int number_of_atoms = 0;
  if (!(input >> number_of_atoms)) {
    PRINT_INPUT_ERROR("Cannot read the atom count from REMD restart xyz.");
  }
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  if (number_of_atoms != source_atom_.number_of_atoms) {
    PRINT_INPUT_ERROR("REMD restart atom count does not match model.");
  }

  std::string comment;
  if (!std::getline(input, comment)) {
    PRINT_INPUT_ERROR("Cannot read the comment line from REMD restart xyz.");
  }

  std::string value;
  if (!extract_quoted_value(comment, "pbc", value)) {
    PRINT_INPUT_ERROR("REMD restart xyz is missing pbc.");
  }
  std::istringstream pbc(value);
  char px, py, pz;
  std::string pbc_trailing;
  if (!(pbc >> px >> py >> pz) || (pbc >> pbc_trailing) ||
      (px != 'T' && px != 't' && px != 'F' && px != 'f') ||
      (py != 'T' && py != 't' && py != 'F' && py != 'f') ||
      (pz != 'T' && pz != 't' && pz != 'F' && pz != 'f')) {
    PRINT_INPUT_ERROR("REMD restart xyz has invalid pbc.");
  }
  logical_replica.box.pbc_x = (px == 'T' || px == 't');
  logical_replica.box.pbc_y = (py == 'T' || py == 't');
  logical_replica.box.pbc_z = (pz == 'T' || pz == 't');

  if (!extract_quoted_value(comment, "Lattice", value)) {
    PRINT_INPUT_ERROR("REMD restart xyz is missing Lattice.");
  }
  std::istringstream lattice(value);
  const int transpose_index[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
  for (int i = 0; i < 9; ++i) {
    if (!(lattice >> logical_replica.box.cpu_h[transpose_index[i]]) ||
        !std::isfinite(logical_replica.box.cpu_h[transpose_index[i]])) {
      PRINT_INPUT_ERROR("REMD restart xyz has invalid Lattice.");
    }
  }
  std::string lattice_trailing;
  if (lattice >> lattice_trailing) {
    PRINT_INPUT_ERROR("REMD restart xyz has invalid Lattice.");
  }
  logical_replica.box.get_inverse();

  if (!extract_token_value(comment, "replica_mode", value) || value != "remd") {
    PRINT_INPUT_ERROR("REMD restart xyz is not an REMD restart file.");
  }
  int restart_replica_id = -1;
  if (!extract_token_value(comment, "replica", value) ||
      !parse_int_value(value, restart_replica_id) || restart_replica_id != replica_id) {
    PRINT_INPUT_ERROR("REMD restart xyz replica id does not match its file name.");
  }
  int temperature_label = -1;
  if (!extract_token_value(comment, "temperature_label", value) ||
      !parse_int_value(value, temperature_label) ||
      temperature_label != replica_to_temperature_[replica_id]) {
    PRINT_INPUT_ERROR("REMD restart xyz temperature label does not match metadata.");
  }
  double target_temperature = 0.0;
  if (!extract_token_value(comment, "target_temperature", value) ||
      !parse_double_value(value, target_temperature) ||
      !values_match(target_temperature, config_.temperatures[temperature_label])) {
    PRINT_INPUT_ERROR("REMD restart xyz target temperature does not match the ladder.");
  }
  double potential_energy = 0.0;
  if (!extract_token_value(comment, "potential_energy", value) ||
      !parse_double_value(value, potential_energy)) {
    PRINT_INPUT_ERROR("REMD restart xyz has invalid potential energy.");
  }
  logical_replica.replica_id = replica_id;
  logical_replica.temperature_label = temperature_label;
  logical_replica.target_temperature = target_temperature;
  logical_replica.potential_energy = potential_energy;

  logical_replica.position_per_atom.resize(number_of_atoms * 3);
  logical_replica.velocity_per_atom.resize(number_of_atoms * 3);
  const double A_per_fs_to_natural = TIME_UNIT_CONVERSION;
  for (int n = 0; n < number_of_atoms; ++n) {
    std::string symbol;
    double mass = 0.0;
    if (!(input >> symbol
          >> logical_replica.position_per_atom[n]
          >> logical_replica.position_per_atom[n + number_of_atoms]
          >> logical_replica.position_per_atom[n + 2 * number_of_atoms]
          >> mass
          >> logical_replica.velocity_per_atom[n]
          >> logical_replica.velocity_per_atom[n + number_of_atoms]
          >> logical_replica.velocity_per_atom[n + 2 * number_of_atoms])) {
      PRINT_INPUT_ERROR("REMD restart xyz atom data is incomplete.");
    }
    if (symbol != source_atom_.cpu_atom_symbol[n] ||
        !values_match(mass, source_atom_.cpu_mass[n]) ||
        !std::isfinite(logical_replica.position_per_atom[n]) ||
        !std::isfinite(logical_replica.position_per_atom[n + number_of_atoms]) ||
        !std::isfinite(logical_replica.position_per_atom[n + 2 * number_of_atoms]) ||
        !std::isfinite(logical_replica.velocity_per_atom[n]) ||
        !std::isfinite(logical_replica.velocity_per_atom[n + number_of_atoms]) ||
        !std::isfinite(logical_replica.velocity_per_atom[n + 2 * number_of_atoms])) {
      PRINT_INPUT_ERROR("REMD restart xyz atom data does not match model.xyz.");
    }
    logical_replica.velocity_per_atom[n] *= A_per_fs_to_natural;
    logical_replica.velocity_per_atom[n + number_of_atoms] *= A_per_fs_to_natural;
    logical_replica.velocity_per_atom[n + 2 * number_of_atoms] *= A_per_fs_to_natural;
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
}

void MultiReplicaDriver::initialize_gpu_slots()
{
  const int max_gpu_slots = config_.devices.size() * config_.replicas_per_gpu;
  const int num_gpu_slots = std::min(config_.replicas, max_gpu_slots);
  if (num_gpu_slots <= 0) {
    PRINT_INPUT_ERROR("No GPU replica slot is available.");
  }
  gpu_slots_.reserve(num_gpu_slots);

  for (int slot_id = 0; slot_id < num_gpu_slots; ++slot_id) {
    std::unique_ptr<ReplicaContext> slot(new ReplicaContext);
    slot->slot_id = slot_id;
    const int device_slot = slot_id % static_cast<int>(config_.devices.size());
    slot->device_id = config_.devices[device_slot];
    initialize_one_gpu_slot(*slot);
    validate_ensemble_support(*slot);
    gpu_slots_.push_back(std::move(slot));
  }

  for (int r = 0; r < config_.replicas; ++r) {
    ReplicaContext& slot = *gpu_slots_[r % num_gpu_slots];
    LogicalReplicaState& logical_replica = logical_replicas_[r];
    CHECK(gpuSetDevice(slot.device_id));
    set_slot_temperature(slot, logical_replica.temperature_label);
    static_cast<Ensemble_BDP*>(slot.integrate.ensemble.get())
      ->seed_rng(config_.seed + 104729 * (r + 1) + 17);
    logical_replica.target_temperature = slot.target_temperature;
    export_ensemble_state(slot, logical_replica);
  }

  if (!config_.resume) {
    exchange_attempts_.assign(std::max(0, config_.replicas - 1), 0);
    exchange_accepts_.assign(std::max(0, config_.replicas - 1), 0);
    exchange_probability_sum_.assign(std::max(0, config_.replicas - 1), 0.0);
  }

  for (int device : config_.devices) {
    CHECK(gpuSetDevice(device));
    CHECK(gpuDeviceSynchronize());
  }

  printf(
    "    active GPU replica slots = %d (%d GPU(s) x %d replicas_per_gpu, capped by %d replicas).\n",
    num_gpu_slots,
    static_cast<int>(config_.devices.size()),
    config_.replicas_per_gpu,
    config_.replicas);
}

void MultiReplicaDriver::validate_ensemble_support(ReplicaContext& replica)
{
  if (replica.integrate.type == 4 && ensemble_command_.size() > 1 &&
      ensemble_command_[1] == "nvt_bdp") {
    return;
  }
  PRINT_INPUT_ERROR("multi_replica remd currently supports only ensemble nvt_bdp.");
}

void MultiReplicaDriver::initialize_one_gpu_slot(ReplicaContext& replica)
{
  CHECK(gpuSetDevice(replica.device_id));

  replica.atom.number_of_atoms = source_atom_.number_of_atoms;
  replica.atom.cpu_type = source_atom_.cpu_type;
  replica.atom.cpu_type_size = source_atom_.cpu_type_size;
  replica.atom.cpu_mass = source_atom_.cpu_mass;
  replica.atom.cpu_charge = source_atom_.cpu_charge;
  replica.atom.cpu_position_per_atom = source_atom_.cpu_position_per_atom;
  replica.atom.cpu_velocity_per_atom = source_atom_.cpu_velocity_per_atom;
  replica.atom.cpu_atom_symbol = source_atom_.cpu_atom_symbol;
  replica.box = source_box_;
  replica.group.resize(source_group_.size());
  for (size_t m = 0; m < source_group_.size(); ++m) {
    replica.group[m].number = source_group_[m].number;
    replica.group[m].cpu_label = source_group_[m].cpu_label;
    replica.group[m].cpu_size = source_group_[m].cpu_size;
    replica.group[m].cpu_size_sum = source_group_[m].cpu_size_sum;
    replica.group[m].cpu_contents = source_group_[m].cpu_contents;
  }

  allocate_memory_gpu(replica.group, replica.atom, replica.thermo);

  for (const auto& command : potential_commands_) {
    std::vector<const char*> potential_param = make_param_array(command);
    replica.force.parse_potential(
      potential_param.data(),
      static_cast<int>(potential_param.size()),
      replica.box,
      replica.atom.type.size(),
      false);
  }

  std::vector<const char*> ensemble_param = make_param_array(ensemble_command_);
  replica.integrate.parse_ensemble(
    ensemble_param.data(),
    static_cast<int>(ensemble_param.size()),
    time_step_,
    replica.atom,
    replica.box,
    replica.group,
    replica.thermo);
  set_slot_temperature(replica, 0);
  replica.integrate.initialize(
    time_step_, replica.atom, replica.box, replica.group, replica.thermo, number_of_steps_);
}

void MultiReplicaDriver::compute_initial_force(ReplicaContext& replica)
{
  CHECK(gpuSetDevice(replica.device_id));
  replica.force.temperature = replica.target_temperature;
  replica.force.delta_T = 0.0;
  replica.force.compute(
    replica.box,
    replica.atom.position_per_atom,
    replica.atom.type,
    replica.group,
    replica.atom.potential_per_atom,
    replica.atom.force_per_atom,
    replica.atom.virial_per_atom,
    replica.atom.velocity_per_atom,
    replica.atom.mass);
}

void MultiReplicaDriver::set_slot_temperature(ReplicaContext& replica, int temperature_label)
{
  replica.target_temperature = get_temperature_from_label(temperature_label);
  replica.integrate.temperature = replica.target_temperature;
  replica.integrate.temperature1 = replica.target_temperature;
  replica.integrate.temperature2 = replica.target_temperature;
  replica.force.temperature = replica.target_temperature;
  replica.force.delta_T = 0.0;
  if (replica.integrate.ensemble) {
    replica.integrate.ensemble->temperature = replica.target_temperature;
  }
}

void MultiReplicaDriver::export_ensemble_state(
  ReplicaContext& slot, LogicalReplicaState& logical_replica)
{
  logical_replica.has_ensemble_state = false;
  logical_replica.ensemble_state.clear();
  if (slot.integrate.type == 4) {
    static_cast<Ensemble_BDP*>(slot.integrate.ensemble.get())
      ->export_state(logical_replica.ensemble_state);
    logical_replica.has_ensemble_state = true;
  }
}

void MultiReplicaDriver::import_ensemble_state(
  LogicalReplicaState& logical_replica, ReplicaContext& slot)
{
  if (!logical_replica.has_ensemble_state) {
    return;
  }
  if (slot.integrate.type == 4) {
    static_cast<Ensemble_BDP*>(slot.integrate.ensemble.get())
      ->import_state(logical_replica.ensemble_state);
  }
}

double MultiReplicaDriver::get_temperature_from_label(int temperature_label) const
{
  return config_.temperatures[temperature_label];
}

void MultiReplicaDriver::load_logical_replica_to_slot(
  LogicalReplicaState& logical_replica, ReplicaContext& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  slot.box = logical_replica.box;
  slot.atom.cpu_position_per_atom = logical_replica.position_per_atom;
  slot.atom.cpu_velocity_per_atom = logical_replica.velocity_per_atom;
  slot.atom.position_per_atom.copy_from_host(logical_replica.position_per_atom.data());
  slot.atom.velocity_per_atom.copy_from_host(logical_replica.velocity_per_atom.data());
  import_ensemble_state(logical_replica, slot);
  set_slot_temperature(slot, logical_replica.temperature_label);
  compute_initial_force(slot);
}

void MultiReplicaDriver::save_slot_to_logical_replica(
  ReplicaContext& slot, LogicalReplicaState& logical_replica)
{
  CHECK(gpuSetDevice(slot.device_id));
  slot.atom.position_per_atom.copy_to_host(logical_replica.position_per_atom.data());
  slot.atom.velocity_per_atom.copy_to_host(logical_replica.velocity_per_atom.data());
  logical_replica.box = slot.box;
  logical_replica.potential_energy = slot.potential_energy;
  export_ensemble_state(slot, logical_replica);
}

void MultiReplicaDriver::randomize_logical_replica_velocity(
  LogicalReplicaState& logical_replica, int seed)
{
  const int number_of_atoms = source_atom_.number_of_atoms;
  if (number_of_atoms <= 0) {
    return;
  }

  double target_temperature = logical_replica.target_temperature;
  if (target_temperature <= 0.0) {
    double kinetic_temperature = 0.0;
    for (int n = 0; n < number_of_atoms; ++n) {
      const double vx = logical_replica.velocity_per_atom[n];
      const double vy = logical_replica.velocity_per_atom[n + number_of_atoms];
      const double vz = logical_replica.velocity_per_atom[n + 2 * number_of_atoms];
      kinetic_temperature += source_atom_.cpu_mass[n] * (vx * vx + vy * vy + vz * vz);
    }
    target_temperature = kinetic_temperature / (3.0 * K_B * number_of_atoms);
  }
  if (target_temperature <= 0.0) {
    return;
  }

  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  for (int n = 0; n < number_of_atoms; ++n) {
    const double sigma = std::sqrt(K_B * target_temperature / source_atom_.cpu_mass[n]);
    logical_replica.velocity_per_atom[n] = sigma * normal(rng);
    logical_replica.velocity_per_atom[n + number_of_atoms] = sigma * normal(rng);
    logical_replica.velocity_per_atom[n + 2 * number_of_atoms] = sigma * normal(rng);
  }

  double total_mass = 0.0;
  double center_of_mass_velocity[3] = {0.0, 0.0, 0.0};
  for (int n = 0; n < number_of_atoms; ++n) {
    const double mass = source_atom_.cpu_mass[n];
    total_mass += mass;
    center_of_mass_velocity[0] += mass * logical_replica.velocity_per_atom[n];
    center_of_mass_velocity[1] += mass * logical_replica.velocity_per_atom[n + number_of_atoms];
    center_of_mass_velocity[2] += mass * logical_replica.velocity_per_atom[n + 2 * number_of_atoms];
  }
  for (int d = 0; d < 3; ++d) {
    center_of_mass_velocity[d] /= total_mass;
  }
  for (int n = 0; n < number_of_atoms; ++n) {
    logical_replica.velocity_per_atom[n] -= center_of_mass_velocity[0];
    logical_replica.velocity_per_atom[n + number_of_atoms] -= center_of_mass_velocity[1];
    logical_replica.velocity_per_atom[n + 2 * number_of_atoms] -= center_of_mass_velocity[2];
  }

  double actual_temperature = 0.0;
  for (int n = 0; n < number_of_atoms; ++n) {
    const double vx = logical_replica.velocity_per_atom[n];
    const double vy = logical_replica.velocity_per_atom[n + number_of_atoms];
    const double vz = logical_replica.velocity_per_atom[n + 2 * number_of_atoms];
    actual_temperature += source_atom_.cpu_mass[n] * (vx * vx + vy * vy + vz * vz);
  }
  actual_temperature /= 3.0 * K_B * number_of_atoms;
  if (actual_temperature > 0.0) {
    scale_logical_replica_velocity(logical_replica, std::sqrt(target_temperature / actual_temperature));
  }
}

void MultiReplicaDriver::scale_logical_replica_velocity(
  LogicalReplicaState& logical_replica, double factor)
{
  const int number_of_atoms = source_atom_.number_of_atoms;
  for (int n = 0; n < number_of_atoms; ++n) {
    logical_replica.velocity_per_atom[n] *= factor;
    logical_replica.velocity_per_atom[n + number_of_atoms] *= factor;
    logical_replica.velocity_per_atom[n + 2 * number_of_atoms] *= factor;
  }
}

void MultiReplicaDriver::update_logical_replica_temperature(
  LogicalReplicaState& logical_replica, int temperature_label)
{
  logical_replica.temperature_label = temperature_label;
  logical_replica.target_temperature = get_temperature_from_label(temperature_label);
}

void MultiReplicaDriver::advance_replica(
  ReplicaContext& replica, int start_step, int steps, int total_steps)
{
  CHECK(gpuSetDevice(replica.device_id));
  const int progress_steps = total_steps > 0 ? total_steps : number_of_steps_;
  for (int s = 0; s < steps; ++s) {
    int step = start_step + s;
    const double run_fraction = progress_steps > 0 ? double(step) / progress_steps : 0.0;
    replica.integrate.current_step = step;
    {
      ScopedTimer timer(replica.timing_compute1);
      replica.integrate.compute1(
        time_step_,
        run_fraction,
        replica.group,
        replica.box,
        replica.atom,
        replica.thermo);
    }

    replica.force.temperature = replica.target_temperature;
    {
      ScopedTimer timer(replica.timing_force);
      replica.force.compute(
        replica.box,
        replica.atom.position_per_atom,
        replica.atom.type,
        replica.group,
        replica.atom.potential_per_atom,
        replica.atom.force_per_atom,
        replica.atom.virial_per_atom,
        replica.atom.velocity_per_atom,
        replica.atom.mass);
    }

    {
      ScopedTimer timer(replica.timing_compute2);
      replica.integrate.compute2(
        time_step_,
        run_fraction,
        replica.group,
        replica.box,
        replica.atom,
        replica.thermo,
        replica.force);
    }
  }

  {
    ScopedTimer timer(replica.timing_thermo_copy);
    double thermo[8];
    replica.thermo.copy_to_host(thermo, 8);
    replica.potential_energy = thermo[1];
  }

}

void MultiReplicaDriver::worker_loop(int slot_id)
{
  WorkerSlot& ws = *workers_[slot_id];
  while (true) {
    std::unique_lock<std::mutex> lock(ws.mtx);
    ws.cv.wait(lock, [&ws] { return ws.ready || ws.stop; });
    if (ws.stop) break;

    ReplicaContext& slot = *gpu_slots_[slot_id];
    LogicalReplicaState& lr = logical_replicas_[ws.logical_id];
    load_logical_replica_to_slot(lr, slot);
    advance_replica(slot, ws.start_step, ws.steps, ws.total_steps);
    save_slot_to_logical_replica(slot, lr);

    ws.ready = false;
    ws.finished = true;
    lock.unlock();
    ws.cv.notify_one();
  }
}

void MultiReplicaDriver::start_workers()
{
  workers_.reserve(gpu_slots_.size());
  for (size_t i = 0; i < gpu_slots_.size(); ++i) {
    workers_.push_back(std::make_unique<WorkerSlot>());
  }
  for (size_t i = 0; i < gpu_slots_.size(); ++i) {
    workers_[i]->thread = std::thread(&MultiReplicaDriver::worker_loop, this, i);
  }
}

void MultiReplicaDriver::stop_workers()
{
  for (auto& ws : workers_) {
    {
      std::lock_guard<std::mutex> lock(ws->mtx);
      ws->stop = true;
    }
    ws->cv.notify_one();
  }
  for (auto& ws : workers_) {
    if (ws->thread.joinable()) ws->thread.join();
  }
}

void MultiReplicaDriver::advance_logical_replicas(
  const std::vector<int>& logical_ids, int start_step, int steps, int total_steps)
{
  const int active_slots = static_cast<int>(gpu_slots_.size());
  for (int batch_start = 0; batch_start < static_cast<int>(logical_ids.size());
       batch_start += active_slots) {
    const int batch_size =
      std::min(active_slots, static_cast<int>(logical_ids.size()) - batch_start);

    for (int slot_id = 0; slot_id < batch_size; ++slot_id) {
      WorkerSlot& ws = *workers_[slot_id];
      std::lock_guard<std::mutex> lock(ws.mtx);
      ws.logical_id = logical_ids[batch_start + slot_id];
      ws.start_step = start_step;
      ws.steps = steps;
      ws.total_steps = total_steps;
      ws.finished = false;
      ws.ready = true;
    }
    for (int slot_id = 0; slot_id < batch_size; ++slot_id) {
      workers_[slot_id]->cv.notify_one();
    }
    for (int slot_id = 0; slot_id < batch_size; ++slot_id) {
      WorkerSlot& ws = *workers_[slot_id];
      std::unique_lock<std::mutex> lock(ws.mtx);
      ws.cv.wait(lock, [&ws] { return ws.finished; });
    }
  }
}

void MultiReplicaDriver::attempt_exchanges(int exchange_count, int step, FILE* exchange_log)
{
  const int parity = exchange_count % 2;
  std::mt19937 rng(config_.seed + exchange_count);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  for (int low_temperature_label = parity; low_temperature_label + 1 < config_.replicas;
       low_temperature_label += 2) {
    const int high_temperature_label = low_temperature_label + 1;
    const int replica_low = temperature_to_replica_[low_temperature_label];
    const int replica_high = temperature_to_replica_[high_temperature_label];
    LogicalReplicaState& low = logical_replicas_[replica_low];
    LogicalReplicaState& high = logical_replicas_[replica_high];

    const double t_low = config_.temperatures[low_temperature_label];
    const double t_high = config_.temperatures[high_temperature_label];
    const double beta_low = config_.betas[low_temperature_label];
    const double beta_high = config_.betas[high_temperature_label];
    const double exchange_energy_low = low.potential_energy;
    const double exchange_energy_high = high.potential_energy;
    const double boltz_factor = (exchange_energy_low - exchange_energy_high) * (beta_low - beta_high);
    const double probability = boltz_factor >= 0.0 ? 1.0 : std::exp(boltz_factor);
    const bool accepted = boltz_factor >= 0.0 || uniform(rng) < probability;
    exchange_attempts_[low_temperature_label] += 1;
    exchange_probability_sum_[low_temperature_label] += probability;

    if (exchange_log) {
      fprintf(
        exchange_log,
        "%d %d %d %d %d %.10e %.10e %.10e %.10e %.10e %.10e %d\n",
        step,
        replica_low,
        replica_high,
        low_temperature_label,
        high_temperature_label,
        low.potential_energy,
        high.potential_energy,
        exchange_energy_low,
        exchange_energy_high,
        boltz_factor,
        probability,
        accepted ? 1 : 0);
    }

    if (!accepted) {
      continue;
    }
    exchange_accepts_[low_temperature_label] += 1;

    scale_logical_replica_velocity(low, std::sqrt(t_high / t_low));
    scale_logical_replica_velocity(high, std::sqrt(t_low / t_high));
    update_logical_replica_temperature(low, high_temperature_label);
    update_logical_replica_temperature(high, low_temperature_label);
    replica_to_temperature_[replica_low] = high_temperature_label;
    replica_to_temperature_[replica_high] = low_temperature_label;
    temperature_to_replica_[low_temperature_label] = replica_high;
    temperature_to_replica_[high_temperature_label] = replica_low;
  }
  if (exchange_log) {
    fflush(exchange_log);
  }
}

void MultiReplicaDriver::run_remd()
{
  if (can_use_resident_remd()) {
    run_remd_resident();
  } else {
    start_workers();
    run_remd_batched();
    stop_workers();
  }
}

bool MultiReplicaDriver::can_use_resident_remd() const
{
  return static_cast<int>(gpu_slots_.size()) == config_.replicas;
}

static int get_remd_equilibration_chunk(int total_steps, int completed_steps)
{
  const int progress_chunk = std::max(1, total_steps / 10);
  return std::min(progress_chunk, total_steps - completed_steps);
}

void MultiReplicaDriver::equilibrate_remd_resident()
{
  if (config_.equilibration_steps <= 0) {
    return;
  }

  printf(
    "    equilibrating each REMD replica for %d steps before production exchanges.\n",
    config_.equilibration_steps);
  fflush(stdout);

  const auto time_begin = std::chrono::high_resolution_clock::now();
  for (int start_step = 0; start_step < config_.equilibration_steps;) {
    const int steps_this_round =
      get_remd_equilibration_chunk(config_.equilibration_steps, start_step);
    {
      ScopedTimer timer(timing_.propagate);
      advance_resident_slots(
        start_step, steps_this_round, config_.equilibration_steps);
    }
    start_step += steps_this_round;
    printf("    %d REMD equilibration steps completed.\n", start_step);
    fflush(stdout);
  }
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  printf("    REMD equilibration finished in %g second.\n", time_used.count());
  fflush(stdout);
}

void MultiReplicaDriver::equilibrate_remd_batched()
{
  if (config_.equilibration_steps <= 0) {
    return;
  }

  printf(
    "    equilibrating each REMD replica for %d steps before production exchanges.\n",
    config_.equilibration_steps);
  fflush(stdout);

  const std::vector<int> logical_ids = make_logical_ids(config_.replicas);
  const auto time_begin = std::chrono::high_resolution_clock::now();
  for (int start_step = 0; start_step < config_.equilibration_steps;) {
    const int steps_this_round =
      get_remd_equilibration_chunk(config_.equilibration_steps, start_step);
    {
      ScopedTimer timer(timing_.propagate);
      advance_logical_replicas(
        logical_ids, start_step, steps_this_round, config_.equilibration_steps);
    }
    start_step += steps_this_round;
    printf("    %d REMD equilibration steps completed.\n", start_step);
    fflush(stdout);
  }
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  printf("    REMD equilibration finished in %g second.\n", time_used.count());
  fflush(stdout);
}

void MultiReplicaDriver::initialize_resident_slots()
{
  for (int r = 0; r < config_.replicas; ++r) {
    ReplicaContext& slot = *gpu_slots_[r];
    LogicalReplicaState& lr = logical_replicas_[r];
    CHECK(gpuSetDevice(slot.device_id));
    load_logical_replica_to_slot(lr, slot);
    // Export initial ensemble state for restart consistency
    export_ensemble_state(slot, lr);
  }
}

void MultiReplicaDriver::advance_resident_slots(int start_step, int steps, int total_steps)
{
  const int N = static_cast<int>(gpu_slots_.size());
  for (int slot_id = 0; slot_id < N; ++slot_id) {
    WorkerSlot& ws = *resident_workers_[slot_id];
    std::lock_guard<std::mutex> lock(ws.mtx);
    ws.start_step = start_step;
    ws.steps = steps;
    ws.total_steps = total_steps;
    ws.finished = false;
    ws.ready = true;
  }
  for (int slot_id = 0; slot_id < N; ++slot_id) {
    resident_workers_[slot_id]->cv.notify_one();
  }
  for (int slot_id = 0; slot_id < N; ++slot_id) {
    WorkerSlot& ws = *resident_workers_[slot_id];
    std::unique_lock<std::mutex> lock(ws.mtx);
    ws.cv.wait(lock, [&ws] { return ws.finished; });
  }
}

void MultiReplicaDriver::resident_worker_loop(int slot_id)
{
  WorkerSlot& ws = *resident_workers_[slot_id];
  while (true) {
    std::unique_lock<std::mutex> lock(ws.mtx);
    ws.cv.wait(lock, [&ws] { return ws.ready || ws.stop; });
    if (ws.stop) break;

    ReplicaContext& slot = *gpu_slots_[slot_id];
    advance_replica(slot, ws.start_step, ws.steps, ws.total_steps);
    LogicalReplicaState& lr = logical_replicas_[slot_id];
    lr.potential_energy = slot.potential_energy;
    lr.box = slot.box;
    export_ensemble_state(slot, lr);

    ws.ready = false;
    ws.finished = true;
    lock.unlock();
    ws.cv.notify_one();
  }
}

void MultiReplicaDriver::start_resident_workers()
{
  resident_workers_.reserve(gpu_slots_.size());
  for (size_t i = 0; i < gpu_slots_.size(); ++i) {
    resident_workers_.push_back(std::make_unique<WorkerSlot>());
  }
  for (size_t i = 0; i < gpu_slots_.size(); ++i) {
    resident_workers_[i]->thread =
      std::thread(&MultiReplicaDriver::resident_worker_loop, this, i);
  }
}

void MultiReplicaDriver::stop_resident_workers()
{
  for (auto& ws : resident_workers_) {
    {
      std::lock_guard<std::mutex> lock(ws->mtx);
      ws->stop = true;
    }
    ws->cv.notify_one();
  }
  for (auto& ws : resident_workers_) {
    if (ws->thread.joinable()) ws->thread.join();
  }
}


// Local GPU kernel for resident-mode velocity scaling
static __global__ void gpu_scale_velocity_kernel(
  const int N, const double factor,
  double* g_vx, double* g_vy, double* g_vz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_vx[i] *= factor;
    g_vy[i] *= factor;
    g_vz[i] *= factor;
  }
}

void MultiReplicaDriver::scale_slot_velocity(ReplicaContext& slot, double factor)
{
  CHECK(gpuSetDevice(slot.device_id));
  const int N = slot.atom.number_of_atoms;
  gpu_scale_velocity_kernel<<<(N - 1) / 128 + 1, 128>>>(
    N, factor,
    slot.atom.velocity_per_atom.data(),
    slot.atom.velocity_per_atom.data() + N,
    slot.atom.velocity_per_atom.data() + 2 * N);
  GPU_CHECK_KERNEL
}

void MultiReplicaDriver::update_resident_replica_temperature(
  int replica_id, int new_label)
{
  ReplicaContext& slot = *gpu_slots_[replica_id];
  LogicalReplicaState& lr = logical_replicas_[replica_id];
  CHECK(gpuSetDevice(slot.device_id));

  const double old_T = lr.target_temperature;
  const double new_T = get_temperature_from_label(new_label);

  if (old_T > 0.0 && new_T > 0.0 && std::fabs(new_T - old_T) > 1e-10) {
    scale_slot_velocity(slot, std::sqrt(new_T / old_T));
  }

  lr.temperature_label = new_label;
  lr.target_temperature = new_T;
  set_slot_temperature(slot, new_label);
}

void MultiReplicaDriver::attempt_exchanges_resident(
  int exchange_count, int step, FILE* exchange_log)
{
  const int parity = exchange_count % 2;
  std::mt19937 rng(config_.seed + exchange_count);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  for (int low_label = parity; low_label + 1 < config_.replicas; low_label += 2) {
    const int high_label = low_label + 1;
    const int rep_low = temperature_to_replica_[low_label];
    const int rep_high = temperature_to_replica_[high_label];
    // In resident mode replica_id == slot_id
    LogicalReplicaState& lr_low = logical_replicas_[rep_low];
    LogicalReplicaState& lr_high = logical_replicas_[rep_high];

    const double t_low = config_.temperatures[low_label];
    const double t_high = config_.temperatures[high_label];
    const double beta_low = config_.betas[low_label];
    const double beta_high = config_.betas[high_label];
    const double exchange_energy_low = lr_low.potential_energy;
    const double exchange_energy_high = lr_high.potential_energy;
    const double boltz_factor =
      (exchange_energy_low - exchange_energy_high) * (beta_low - beta_high);
    const double probability = boltz_factor >= 0.0 ? 1.0 : std::exp(boltz_factor);
    const bool accepted = boltz_factor >= 0.0 || uniform(rng) < probability;
    exchange_attempts_[low_label] += 1;
    exchange_probability_sum_[low_label] += probability;

    if (exchange_log) {
      fprintf(
        exchange_log,
        "%d %d %d %d %d %.10e %.10e %.10e %.10e %.10e %.10e %d\n",
        step, rep_low, rep_high, low_label, high_label,
        lr_low.potential_energy, lr_high.potential_energy,
        exchange_energy_low, exchange_energy_high,
        boltz_factor, probability, accepted ? 1 : 0);
    }

    if (!accepted) continue;
    exchange_accepts_[low_label] += 1;

    // Scale velocities directly on GPU slots
    update_resident_replica_temperature(rep_low, high_label);
    update_resident_replica_temperature(rep_high, low_label);

    replica_to_temperature_[rep_low] = high_label;
    replica_to_temperature_[rep_high] = low_label;
    temperature_to_replica_[low_label] = rep_high;
    temperature_to_replica_[high_label] = rep_low;
  }
  if (exchange_log) {
    fflush(exchange_log);
  }
}

void MultiReplicaDriver::open_remd_logs(
  FILE*& exchange_log, FILE*& status_log, FILE*& thermo_log) const
{
  exchange_log = config_.verbose_output ? my_fopen("remd_exchange.out", "w") : nullptr;
  status_log = (config_.verbose_output || config_.dump_xyz)
                 ? my_fopen("remd_temperature_status.out", "w")
                 : nullptr;
  thermo_log = config_.verbose_output ? my_fopen("remd_thermo.out", "w") : nullptr;

  if (exchange_log) {
    fprintf(
      exchange_log,
      "# step replica_i replica_j temp_i temp_j pe_i pe_j exchange_energy_i exchange_energy_j "
      "boltz_factor probability accepted\n");
  }
  if (status_log) {
    fprintf(status_log, "# step  replica_of_temp_0 ... temp_%d\n", config_.replicas - 1);
  }
  if (thermo_log) {
    fprintf(thermo_log, "# step");
    for (int r = 0; r < config_.replicas; ++r) {
      fprintf(thermo_log, " target_T_%d PE_%d Vol_%d", r, r, r);
    }
    fprintf(thermo_log, "\n");
  }
}

void MultiReplicaDriver::close_remd_logs(
  FILE* exchange_log, FILE* status_log, FILE* thermo_log) const
{
  if (exchange_log) fclose(exchange_log);
  if (status_log) fclose(status_log);
  if (thermo_log) fclose(thermo_log);
}

void MultiReplicaDriver::write_remd_thermo(int step, FILE* thermo_log) const
{
  if (!thermo_log) {
    return;
  }
  fprintf(thermo_log, "%d", step);
  for (int r = 0; r < config_.replicas; ++r) {
    fprintf(thermo_log, " %.6g %.6g %.6g",
            logical_replicas_[r].target_temperature,
            logical_replicas_[r].potential_energy,
            logical_replicas_[r].box.get_volume());
  }
  fprintf(thermo_log, "\n");
  fflush(thermo_log);
}

void MultiReplicaDriver::print_remd_progress(int step) const
{
  const int base = (10 <= number_of_steps_) ? (number_of_steps_ / 10) : 1;
  if (0 == step % base || step == number_of_steps_) {
    printf("    %d REMD steps completed.\n", step);
    fflush(stdout);
  }
}

void MultiReplicaDriver::print_remd_performance(double time_used) const
{
  print_line_1();
  printf("Time used for this REMD run = %g second.\n", time_used);
  const double run_speed =
    source_atom_.number_of_atoms * (number_of_steps_ * config_.replicas / time_used);
  printf("Speed of this REMD run = %g atom*step/second.\n", run_speed);
  print_line_2();
}

void MultiReplicaDriver::write_remd_outputs(FILE* exchange_log)
{
  ScopedTimer timer(timing_.output);
  write_exchange_statistics(exchange_log);
  write_ladder_report();
}

void MultiReplicaDriver::run_remd_resident()
{
  FILE* exchange_log = nullptr;
  FILE* status_log = nullptr;
  FILE* thermo_log = nullptr;
  open_remd_logs(exchange_log, status_log, thermo_log);
  open_remd_xyz_dump();

  {
    ScopedTimer timer(timing_.initialize);
    initialize_resident_slots();
    start_resident_workers();
  }

  equilibrate_remd_resident();

  write_temperature_ladder_file();
  write_temperature_status(0, status_log);

  const auto time_begin = std::chrono::high_resolution_clock::now();
  for (int start_step = 0; start_step < number_of_steps_;) {
    int next_event_step = std::min(
      number_of_steps_,
      ((start_step / config_.exchange_interval) + 1) * config_.exchange_interval);
    if (config_.dump_xyz) {
      next_event_step = std::min(
        next_event_step,
        ((start_step / config_.dump_xyz_interval) + 1) * config_.dump_xyz_interval);
    }
    const int steps_this_round = next_event_step - start_step;

    {
      ScopedTimer timer(timing_.propagate);
      advance_resident_slots(start_step, steps_this_round);
    }
    start_step += steps_this_round;

    write_remd_thermo(start_step, thermo_log);

    const bool exchange_due = start_step % config_.exchange_interval == 0;
    const bool dump_due =
      config_.dump_xyz && start_step % config_.dump_xyz_interval == 0;

    if (exchange_due) {
      {
        ScopedTimer timer(timing_.exchange);
        attempt_exchanges_resident(remd_exchange_count_, start_step, exchange_log);
      }
      ++remd_exchange_count_;
    }

    if (dump_due) {
      write_temperature_status(start_step, status_log);
      write_remd_xyz_dump(start_step, true);
    } else if (!config_.dump_xyz && exchange_due) {
      write_temperature_status(start_step, status_log);
    }

    print_remd_progress(start_step);
  }

  stop_resident_workers();
  {
    ScopedTimer timer(timing_.restart);
    // Full save for restart
    for (int r = 0; r < config_.replicas; ++r) {
      ReplicaContext& slot = *gpu_slots_[r];
      LogicalReplicaState& lr = logical_replicas_[r];
      save_slot_to_logical_replica(slot, lr);
    }
    write_restart_files("remd");
    write_remd_restart_metadata("remd");
  }

  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  print_remd_performance(time_used.count());

  write_remd_outputs(exchange_log);
  close_remd_xyz_dump();
  close_remd_logs(exchange_log, status_log, thermo_log);
}

void MultiReplicaDriver::run_remd_batched()
{
  FILE* exchange_log = nullptr;
  FILE* status_log = nullptr;
  FILE* thermo_log = nullptr;
  open_remd_logs(exchange_log, status_log, thermo_log);
  open_remd_xyz_dump();

  equilibrate_remd_batched();

  write_temperature_ladder_file();
  write_temperature_status(0, status_log);

  const auto time_begin = std::chrono::high_resolution_clock::now();
  const std::vector<int> logical_ids = make_logical_ids(config_.replicas);
  for (int start_step = 0; start_step < number_of_steps_;) {
    int next_event_step = std::min(
      number_of_steps_,
      ((start_step / config_.exchange_interval) + 1) * config_.exchange_interval);
    if (config_.dump_xyz) {
      next_event_step = std::min(
        next_event_step,
        ((start_step / config_.dump_xyz_interval) + 1) * config_.dump_xyz_interval);
    }
    const int steps_this_round = next_event_step - start_step;

    {
      ScopedTimer timer(timing_.propagate);
      advance_logical_replicas(logical_ids, start_step, steps_this_round);
    }

    start_step += steps_this_round;

    write_remd_thermo(start_step, thermo_log);

    const bool exchange_due = start_step % config_.exchange_interval == 0;
    const bool dump_due =
      config_.dump_xyz && start_step % config_.dump_xyz_interval == 0;

    if (exchange_due) {
      {
        ScopedTimer timer(timing_.exchange);
        attempt_exchanges(remd_exchange_count_, start_step, exchange_log);
      }
      ++remd_exchange_count_;
    }

    if (dump_due) {
      write_temperature_status(start_step, status_log);
      write_remd_xyz_dump(start_step, false);
    } else if (!config_.dump_xyz && exchange_due) {
      write_temperature_status(start_step, status_log);
    }

    print_remd_progress(start_step);
  }

  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;
  print_remd_performance(time_used.count());

  write_remd_outputs(exchange_log);
  {
    ScopedTimer timer(timing_.restart);
    write_restart_files("remd");
    write_remd_restart_metadata("remd");
  }
  close_remd_xyz_dump();
  close_remd_logs(exchange_log, status_log, thermo_log);
}

void MultiReplicaDriver::write_exchange_statistics(FILE* exchange_log)
{
  if (!exchange_log) {
    return;
  }
  fprintf(exchange_log, "\n# REMD exchange statistics by neighboring temperature label\n");
  fprintf(exchange_log, "# temp_i temp_j attempts accepted acceptance_ratio average_probability\n");
  for (int i = 0; i + 1 < config_.replicas; ++i) {
    const int attempts = exchange_attempts_[i];
    const int accepts = exchange_accepts_[i];
    const double acceptance_ratio = attempts > 0 ? double(accepts) / attempts : 0.0;
    const double average_probability =
      attempts > 0 ? exchange_probability_sum_[i] / attempts : 0.0;
    fprintf(
      exchange_log,
      "%d %d %d %d %.10e %.10e\n",
      i,
      i + 1,
      attempts,
      accepts,
      acceptance_ratio,
      average_probability);
  }
  fflush(exchange_log);
}

void MultiReplicaDriver::write_temperature_ladder_file()
{
  FILE* ladder_file = my_fopen("remd_temps.in", "w");
  fprintf(ladder_file, "# REMD temperature ladder generated by GPUMD\n");
  fprintf(ladder_file, "# replicas %d\n", config_.replicas);
  if (config_.generated_temperatures) {
    fprintf(
      ladder_file,
      "# temp %.16g %.16g\n",
      config_.temperature_min,
      config_.temperature_max);
    fprintf(
      ladder_file,
      "# spacing %s\n",
      config_.temperature_style == MultiReplicaConfig::geometric
        ? "geometric"
        : "linear");
  }
  for (int r = 0; r < config_.replicas; ++r) {
    fprintf(ladder_file, "%.16g\n", config_.temperatures[r]);
  }
  fflush(ladder_file);
  fclose(ladder_file);
}

void MultiReplicaDriver::write_ladder_report()
{
  const std::string report_name = "remd_ladder.out";
  FILE* report = my_fopen(report_name.c_str(), "w");
  fprintf(report, "# REMD temperature ladder report\n");
  fprintf(report, "# target_acceptance %.10e\n", DEFAULT_REMD_TARGET_ACCEPTANCE);
  fprintf(
    report,
    "# temp_i temp_j temperature_i temperature_j attempts accepted acceptance_ratio "
    "average_probability meets_target\n");

  bool all_met_target = true;
  for (int i = 0; i + 1 < config_.replicas; ++i) {
    const int attempts = exchange_attempts_[i];
    const int accepts = exchange_accepts_[i];
    const double acceptance_ratio = attempts > 0 ? double(accepts) / attempts : 0.0;
    const double average_probability =
      attempts > 0 ? exchange_probability_sum_[i] / attempts : 0.0;
    const bool meets_target = attempts > 0 && acceptance_ratio >= DEFAULT_REMD_TARGET_ACCEPTANCE;
    all_met_target = all_met_target && meets_target;
    fprintf(
      report,
      "%d %d %.16g %.16g %d %d %.10e %.10e %d\n",
      i,
      i + 1,
      config_.temperatures[i],
      config_.temperatures[i + 1],
      attempts,
      accepts,
      acceptance_ratio,
      average_probability,
      meets_target ? 1 : 0);
  }

  fprintf(
    report,
    "# summary all_pairs_meet_target %d\n",
    all_met_target ? 1 : 0);
  fflush(report);
  fclose(report);

  printf(
    "    REMD ladder report written to %s (%s target acceptance %g).\n",
    report_name.c_str(),
    all_met_target ? "all pairs meet" : "some pairs are below",
    DEFAULT_REMD_TARGET_ACCEPTANCE);
}

void MultiReplicaDriver::write_temperature_status(int step, FILE* status_log)
{
  if (!status_log) {
    return;
  }
  fprintf(status_log, "%d", step);
  for (int temperature_label = 0; temperature_label < config_.replicas; ++temperature_label) {
    fprintf(status_log, " %d", temperature_to_replica_[temperature_label]);
  }
  fprintf(status_log, "\n");
  fflush(status_log);
}

static std::string get_remd_dump_filename(const std::string& prefix, int replica_id)
{
  const std::string replica_suffix = "_replica_" + std::to_string(replica_id);
  const size_t path_separator = prefix.find_last_of("/\\");
  const size_t extension = prefix.find_last_of('.');
  if (extension != std::string::npos &&
      (path_separator == std::string::npos || extension > path_separator)) {
    return prefix.substr(0, extension) + replica_suffix + prefix.substr(extension);
  }
  return prefix + replica_suffix + ".xyz";
}

void MultiReplicaDriver::open_remd_xyz_dump()
{
  if (!config_.dump_xyz) {
    return;
  }
  remd_dump_xyz_files_.resize(config_.replicas, nullptr);
  for (int r = 0; r < config_.replicas; ++r) {
    const std::string filename = get_remd_dump_filename(config_.dump_xyz_filename, r);
    remd_dump_xyz_files_[r] = my_fopen(filename.c_str(), "w");
  }
  const int number_of_atoms = source_atom_.number_of_atoms;
  remd_dump_position_.resize(number_of_atoms * 3);
  if (config_.dump_xyz_has_velocity) {
    remd_dump_velocity_.resize(number_of_atoms * 3);
  }
  printf(
    "    REMD dump_xyz: write %d replica trajectories with prefix %s every %d steps%s.\n",
    config_.replicas,
    config_.dump_xyz_filename.c_str(),
    config_.dump_xyz_interval,
    config_.dump_xyz_has_velocity ? " with velocities" : "");
  fflush(stdout);
}

void MultiReplicaDriver::close_remd_xyz_dump()
{
  for (FILE* fid : remd_dump_xyz_files_) {
    if (fid) {
      fclose(fid);
    }
  }
  remd_dump_xyz_files_.clear();
}

static void write_remd_xyz_line2(
  FILE* fid,
  const Box& box,
  double time_step,
  int step,
  int replica_id,
  int temperature_label,
  double target_temperature,
  bool has_velocity)
{
  fprintf(
    fid,
    "Time=%.8f step=%d replica=%d temperature_label=%d target_temperature=%.16g ",
    step * time_step * TIME_UNIT_CONVERSION,
    step,
    replica_id,
    temperature_label,
    target_temperature);
  fprintf(
    fid,
    "pbc=\"%c %c %c\" ",
    box.pbc_x ? 'T' : 'F',
    box.pbc_y ? 'T' : 'F',
    box.pbc_z ? 'T' : 'F');
  fprintf(
    fid,
    "Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\" ",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);
  fprintf(fid, "Properties=species:S:1:pos:R:3");
  if (has_velocity) {
    fprintf(fid, ":vel:R:3");
  }
  fprintf(fid, "\n");
}

void MultiReplicaDriver::write_remd_xyz_dump(int step, bool resident_mode)
{
  if (!config_.dump_xyz || step <= 0 || step % config_.dump_xyz_interval != 0) {
    return;
  }

  const int number_of_atoms = source_atom_.number_of_atoms;
  const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;

  if (static_cast<int>(remd_dump_xyz_files_.size()) != config_.replicas) {
    return;
  }
  for (int r = 0; r < config_.replicas; ++r) {
    FILE* fid = remd_dump_xyz_files_[r];
    if (!fid) {
      continue;
    }

    const LogicalReplicaState& logical_replica = logical_replicas_[r];
    const std::vector<double>* position = &logical_replica.position_per_atom;
    const std::vector<double>* velocity = &logical_replica.velocity_per_atom;
    Box box = logical_replica.box;

    if (resident_mode) {
      ReplicaContext& slot = *gpu_slots_[r];
      CHECK(gpuSetDevice(slot.device_id));
      slot.atom.position_per_atom.copy_to_host(remd_dump_position_.data());
      if (config_.dump_xyz_has_velocity) {
        slot.atom.velocity_per_atom.copy_to_host(remd_dump_velocity_.data());
      }
      position = &remd_dump_position_;
      velocity = &remd_dump_velocity_;
      box = slot.box;
    }

    fprintf(fid, "%d\n", number_of_atoms);
    write_remd_xyz_line2(
      fid,
      box,
      time_step_,
      step,
      r,
      logical_replica.temperature_label,
      logical_replica.target_temperature,
      config_.dump_xyz_has_velocity != 0);

    for (int n = 0; n < number_of_atoms; ++n) {
      fprintf(
        fid,
        "%s %.8f %.8f %.8f",
        source_atom_.cpu_atom_symbol[n].c_str(),
        (*position)[n],
        (*position)[n + number_of_atoms],
        (*position)[n + 2 * number_of_atoms]);
      if (config_.dump_xyz_has_velocity) {
        fprintf(
          fid,
          " %.8f %.8f %.8f",
          (*velocity)[n] * natural_to_A_per_fs,
          (*velocity)[n + number_of_atoms] * natural_to_A_per_fs,
          (*velocity)[n + 2 * number_of_atoms] * natural_to_A_per_fs);
      }
      fprintf(fid, "\n");
    }
    fflush(fid);
  }
}

void MultiReplicaDriver::write_restart_files(const char* mode_name)
{
  const int number_of_atoms = source_atom_.number_of_atoms;
  const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;

  for (int r = 0; r < config_.replicas; ++r) {
    const LogicalReplicaState& logical_replica = logical_replicas_[r];
    const std::string filename =
      std::string(mode_name) + "_replica_" + std::to_string(r) + "_restart.xyz";
    FILE* fid = my_fopen(filename.c_str(), "w");

    fprintf(fid, "%d\n", number_of_atoms);
    fprintf(
      fid,
      "pbc=\"%c %c %c\" ",
      logical_replica.box.pbc_x ? 'T' : 'F',
      logical_replica.box.pbc_y ? 'T' : 'F',
      logical_replica.box.pbc_z ? 'T' : 'F');
    fprintf(
      fid,
      "Lattice=\"%.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g\" ",
      logical_replica.box.cpu_h[0],
      logical_replica.box.cpu_h[3],
      logical_replica.box.cpu_h[6],
      logical_replica.box.cpu_h[1],
      logical_replica.box.cpu_h[4],
      logical_replica.box.cpu_h[7],
      logical_replica.box.cpu_h[2],
      logical_replica.box.cpu_h[5],
      logical_replica.box.cpu_h[8]);
    if (source_group_.empty()) {
      fprintf(fid, "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3 ");
    } else {
      fprintf(
        fid,
        "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:%d ",
        static_cast<int>(source_group_.size()));
    }
    fprintf(
      fid,
      "replica_mode=%s replica=%d temperature_label=%d target_temperature=%.16g "
      "potential_energy=%.16g\n",
      mode_name,
      r,
      logical_replica.temperature_label,
      logical_replica.target_temperature,
      logical_replica.potential_energy);

    for (int n = 0; n < number_of_atoms; ++n) {
      fprintf(
        fid,
        "%s %.16g %.16g %.16g %.16g %.16g %.16g %.16g",
        source_atom_.cpu_atom_symbol[n].c_str(),
        logical_replica.position_per_atom[n],
        logical_replica.position_per_atom[n + number_of_atoms],
        logical_replica.position_per_atom[n + 2 * number_of_atoms],
        source_atom_.cpu_mass[n],
        logical_replica.velocity_per_atom[n] * natural_to_A_per_fs,
        logical_replica.velocity_per_atom[n + number_of_atoms] * natural_to_A_per_fs,
        logical_replica.velocity_per_atom[n + 2 * number_of_atoms] * natural_to_A_per_fs);
      for (size_t m = 0; m < source_group_.size(); ++m) {
        fprintf(fid, " %d", source_group_[m].cpu_label[n]);
      }
      fprintf(fid, "\n");
    }

    fflush(fid);
    fclose(fid);
  }
}

void MultiReplicaDriver::write_remd_restart_metadata(const char* mode_name)
{
  const std::string filename = std::string(mode_name) + "_restart.meta";
  FILE* fid = my_fopen(filename.c_str(), "w");

  fprintf(fid, "# GPUMD REMD restart metadata\n");
  fprintf(fid, "format_version 1\n");
  fprintf(fid, "mode remd\n");
  fprintf(fid, "replicas %d\n", config_.replicas);
  fprintf(fid, "exchange_count %d\n", remd_exchange_count_);

  fprintf(fid, "replica_to_temperature");
  for (int r = 0; r < config_.replicas; ++r) {
    fprintf(fid, " %d", replica_to_temperature_[r]);
  }
  fprintf(fid, "\n");

  fprintf(fid, "temperature_to_replica");
  for (int r = 0; r < config_.replicas; ++r) {
    fprintf(fid, " %d", temperature_to_replica_[r]);
  }
  fprintf(fid, "\n");

  fprintf(fid, "exchange_attempts");
  for (int i = 0; i + 1 < config_.replicas; ++i) {
    fprintf(fid, " %d", exchange_attempts_[i]);
  }
  fprintf(fid, "\n");

  fprintf(fid, "exchange_accepts");
  for (int i = 0; i + 1 < config_.replicas; ++i) {
    fprintf(fid, " %d", exchange_accepts_[i]);
  }
  fprintf(fid, "\n");

  fprintf(fid, "exchange_probability_sum");
  for (int i = 0; i + 1 < config_.replicas; ++i) {
    fprintf(fid, " %.16g", exchange_probability_sum_[i]);
  }
  fprintf(fid, "\n");

  fflush(fid);
  fclose(fid);
}

void MultiReplicaDriver::print_timing_summary(const char* mode_name) const
{
  double sum_compute1 = 0.0;
  double sum_force = 0.0;
  double sum_compute2 = 0.0;
  double sum_thermo = 0.0;
  for (const auto& slot : gpu_slots_) {
    sum_compute1 += slot->timing_compute1;
    sum_force += slot->timing_force;
    sum_compute2 += slot->timing_compute2;
    sum_thermo += slot->timing_thermo_copy;
  }

  printf("Timing summary for %s:\n", mode_name);
  printf("    initialize: %g s\n", timing_.initialize);
  printf("    propagate (wall)     : %g s\n", timing_.propagate);
  printf("    propagate slot-sum   :\n");
  printf("      compute1 : %g s\n", sum_compute1);
  printf("      force    : %g s\n", sum_force);
  printf("      compute2 : %g s\n", sum_compute2);
  printf("      thermo   : %g s\n", sum_thermo);
  printf("    exchange  : %g s\n", timing_.exchange);
  printf("    output    : %g s\n", timing_.output);
  printf("    restart   : %g s\n", timing_.restart);
  printf("    total     : %g s\n", timing_.total);
  fflush(stdout);
}

namespace replica {

struct ReplicaApiState
{
  MultiReplicaConfig config;
  std::vector<std::vector<std::string>> potential_commands;
  std::vector<std::string> ensemble_command;
};

static ReplicaApiState& state()
{
  static ReplicaApiState api_state;
  return api_state;
}

void reset()
{
  state() = ReplicaApiState();
}

bool enabled()
{
  return state().config.enabled();
}

void parse(const char** param, int num_param)
{
  state().config.parse(param, num_param);
}

void remember_potential_command(const std::vector<std::string>& tokens)
{
  state().potential_commands.push_back(tokens);
}

void remember_ensemble_command(const std::vector<std::string>& tokens)
{
  state().ensemble_command = tokens;
}

void remember_dump_xyz_command(const std::vector<std::string>& tokens)
{
  std::vector<const char*> param(tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    param[i] = tokens[i].c_str();
  }
  state().config.parse_dump_xyz(param.data(), static_cast<int>(param.size()));
}

void run(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  double time_step,
  int number_of_steps)
{
  ReplicaApiState& api_state = state();

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());

  MultiReplicaDriver driver(
    api_state.config,
    atom,
    box,
    group,
    api_state.potential_commands,
    api_state.ensemble_command,
    time_step,
    number_of_steps);
  driver.run();
  reset();
}

bool run_if_enabled(
  Atom& atom,
  Box& box,
  std::vector<Group>& group,
  double time_step,
  int number_of_steps,
  double max_distance_per_step)
{
  if (!enabled()) {
    return false;
  }
  if (max_distance_per_step > 0.0) {
    PRINT_INPUT_ERROR("multi_replica does not support adaptive time_step in this version.");
  }
  run(atom, box, group, time_step, number_of_steps);
  return true;
}

} // namespace replica
