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

#include "replica_common.cuh"
#include "replica_batch.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace replica
{

bool values_match(const double first, const double second)
{
  const double scale = std::max(1.0, std::max(std::abs(first), std::abs(second)));
  return std::abs(first - second) <= 1.0e-12 * scale;
}

bool file_has_content(const std::string& filename)
{
  std::ifstream input(filename, std::ios::binary | std::ios::ate);
  return input.is_open() && input.tellg() > std::streampos(0);
}

std::string trim_left(const std::string& value)
{
  const size_t first = value.find_first_not_of(" \t\r\n");
  return first == std::string::npos ? std::string() : value.substr(first);
}

unsigned long long hash_file(const std::string& filename)
{
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the multi_replica potential file.");

  unsigned long long hash = 1469598103934665603ULL;
  char buffer[8192];
  while (input.good()) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize count = input.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

bool extract_quoted_value(
  const std::string& line, const std::string& key, std::string& value)
{
  const std::string needle = key + "=\"";
  const size_t start = line.find(needle);
  if (start == std::string::npos)
    return false;
  const size_t value_start = start + needle.size();
  const size_t value_end = line.find('"', value_start);
  if (value_end == std::string::npos)
    return false;
  value = line.substr(value_start, value_end - value_start);
  return true;
}

bool extract_token_value(
  const std::string& line, const std::string& key, std::string& value)
{
  const std::string needle = key + "=";
  const size_t start = line.find(needle);
  if (start == std::string::npos)
    return false;
  const size_t value_start = start + needle.size();
  const size_t value_end = line.find_first_of(" \t\r\n", value_start);
  value = line.substr(
    value_start,
    value_end == std::string::npos ? std::string::npos : value_end - value_start);
  return true;
}

void validate_xyz_box(
  const std::string& comment, const Box& reference, const char* mode)
{
  std::string value;
  if (!extract_quoted_value(comment, "pbc", value)) {
    const std::string message = std::string(mode) + " restart xyz is missing pbc.";
    PRINT_INPUT_ERROR(message.c_str());
  }
  std::istringstream pbc(value);
  char pbc_x = 0;
  char pbc_y = 0;
  char pbc_z = 0;
  std::string trailing;
  if (!(pbc >> pbc_x >> pbc_y >> pbc_z) || (pbc >> trailing) ||
      pbc_x != (reference.pbc_x ? 'T' : 'F') ||
      pbc_y != (reference.pbc_y ? 'T' : 'F') ||
      pbc_z != (reference.pbc_z ? 'T' : 'F')) {
    const std::string message =
      std::string(mode) + " restart xyz pbc does not match model.xyz.";
    PRINT_INPUT_ERROR(message.c_str());
  }

  if (!extract_quoted_value(comment, "Lattice", value)) {
    const std::string message = std::string(mode) + " restart xyz is missing Lattice.";
    PRINT_INPUT_ERROR(message.c_str());
  }
  std::istringstream lattice(value);
  const int transpose[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
  for (int i = 0; i < 9; ++i) {
    double box_value = 0.0;
    if (!(lattice >> box_value) || !std::isfinite(box_value) ||
        !values_match(box_value, reference.cpu_h[transpose[i]])) {
      const std::string message =
        std::string(mode) + " restart xyz Lattice does not match model.xyz.";
      PRINT_INPUT_ERROR(message.c_str());
    }
  }
  if (lattice >> trailing) {
    const std::string message =
      std::string(mode) + " restart xyz has too many Lattice values.";
    PRINT_INPUT_ERROR(message.c_str());
  }
}

const char RESTART_XYZ_PROPERTIES[] = "species:S:1:pos:R:3:mass:R:1:vel:R:3:vel_natural:R:3";

bool parse_restart_xyz_properties(const std::string& properties, bool& natural_velocity)
{
  natural_velocity = properties == RESTART_XYZ_PROPERTIES;
  return natural_velocity || properties == "species:S:1:pos:R:3:mass:R:1:vel:R:3";
}

void write_restart_xyz_velocity(FILE* file, const std::vector<double>& velocity, const int atom)
{
  const size_t n = velocity.size() / 3;
  fprintf(
    file,
    " %.17g %.17g %.17g %.17g %.17g %.17g\n",
    velocity[atom] / TIME_UNIT_CONVERSION,
    velocity[atom + n] / TIME_UNIT_CONVERSION,
    velocity[atom + 2 * n] / TIME_UNIT_CONVERSION,
    velocity[atom],
    velocity[atom + n],
    velocity[atom + 2 * n]);
}

bool read_restart_xyz_velocity(
  std::istream& input, const bool natural_velocity, std::vector<double>& velocity, const int atom)
{
  const size_t n = velocity.size() / 3;
  double velocity_in_fs[3];
  if (!(input >> velocity_in_fs[0] >> velocity_in_fs[1] >> velocity_in_fs[2]))
    return false;
  if (natural_velocity)
    return static_cast<bool>(input >> velocity[atom] >> velocity[atom + n] >> velocity[atom + 2 * n]);
  for (int d = 0; d < 3; ++d)
    velocity[atom + d * n] = velocity_in_fs[d] * TIME_UNIT_CONVERSION;
  return true;
}

namespace
{

void mode_error(const char* mode, const std::string& text)
{
  const std::string message = std::string(mode) + " " + text;
  PRINT_INPUT_ERROR(message.c_str());
}

} // namespace

void parse_replicas_option(
  const std::vector<std::string>& command,
  size_t& i,
  const int minimum_replicas,
  const char* mode,
  int& replicas,
  int& replicas_per_gpu)
{
  if (++i >= command.size() || !is_valid_int(command[i].c_str(), &replicas) ||
      replicas < minimum_replicas)
    mode_error(
      mode, "replicas should be an integer of at least " + std::to_string(minimum_replicas) + ".");
  ++i;
  int value = 0;
  if (i < command.size() && is_valid_int(command[i].c_str(), &value)) {
    if (value <= 0)
      mode_error(mode, "replicas_per_gpu should be positive.");
    replicas_per_gpu = value;
    ++i;
  }
}

void parse_dump_xyz(
  const std::vector<std::string>& dump_command,
  const char* mode,
  const bool allow_velocity,
  int& dump_interval,
  std::string& dump_prefix,
  bool& dump_velocity)
{
  if (dump_command.size() < 5 || dump_command[0] != "dump_xyz")
    mode_error(mode, "dump_xyz command is invalid.");
  int grouping_method = 0;
  int group_id = 0;
  if (!is_valid_int(dump_command[1].c_str(), &grouping_method) || grouping_method != -1 ||
      !is_valid_int(dump_command[2].c_str(), &group_id))
    mode_error(mode, "dump_xyz requires grouping_method -1.");
  if (!is_valid_int(dump_command[3].c_str(), &dump_interval) || dump_interval <= 0)
    mode_error(mode, "dump_xyz interval should be positive.");
  dump_prefix = dump_command[4];
  if (dump_prefix.empty() || dump_prefix.back() == '*')
    mode_error(mode, "dump_xyz does not support an empty or '*' filename.");
  dump_velocity = false;
  for (size_t i = 5; i < dump_command.size(); ++i) {
    if (allow_velocity && dump_command[i] == "velocity")
      dump_velocity = true;
    else
      mode_error(
        mode,
        allow_velocity ? "dump_xyz supports positions and optional velocity only."
                       : "dump_xyz supports positions only.");
  }
}

unsigned int draw_internal_seed()
{
  const unsigned int seed = static_cast<unsigned int>(rand());
  return seed == 0 ? 1 : seed;
}

unsigned long long validate_potential(
  const std::vector<std::vector<std::string>>& potential_commands,
  const Box& box,
  const char* mode,
  std::string& kspace_method)
{
  if (potential_commands.size() != 1 || potential_commands[0].size() != 2)
    mode_error(mode, "currently requires exactly one standard NEP potential command.");
  if (is_nep_charge_potential(potential_commands[0][1])) {
    if (!box.pbc_x || !box.pbc_y || !box.pbc_z)
      mode_error(
        "qNEP", std::string(mode) + " requires periodic boundaries in all directions.");
  } else {
    kspace_method = "none";
  }
  return hash_file(potential_commands[0][1]);
}

void parse_nvt_bdp(
  const std::vector<std::string>& ensemble_command,
  const char* mode,
  double& temperature_1,
  double& temperature_2,
  double& temperature_coupling)
{
  if (ensemble_command.size() != 5 || ensemble_command[0] != "ensemble" ||
      ensemble_command[1] != "nvt_bdp" ||
      !is_valid_real(ensemble_command[2].c_str(), &temperature_1) ||
      !is_valid_real(ensemble_command[3].c_str(), &temperature_2) || temperature_1 <= 0.0 ||
      temperature_2 <= 0.0 ||
      !is_valid_real(ensemble_command[4].c_str(), &temperature_coupling) ||
      temperature_coupling < 1.0)
    mode_error(mode, "requires ensemble nvt_bdp T1 T2 tau_T with T1, T2 > 0 and tau_T >= 1.");
}

int resolve_replicas_per_gpu(const int replicas, const int replicas_per_gpu, const char* mode)
{
  int number_of_devices = 0;
  CHECK(gpuGetDeviceCount(&number_of_devices));
  if (number_of_devices <= 0)
    mode_error(mode, "found no GPU device.");
  const int result =
    replicas_per_gpu > 0 ? replicas_per_gpu : (replicas + number_of_devices - 1) / number_of_devices;
  if (result * number_of_devices < replicas)
    mode_error(mode, "replicas_per_gpu is too small for the visible GPUs.");
  return result;
}

void initialize_thermostat_rng(
  Replica_Slot& slot,
  const bool resume,
  const std::string& restart_state,
  const unsigned int internal_seed,
  const char* mode)
{
  if (resume) {
    if (!slot.thermostat->import_rng_state(restart_state))
      mode_error(mode, "restart has an invalid nvt_bdp RNG state.");
  } else {
    slot.thermostat->seed_rng(internal_seed + 130363U * (slot.replica_id + 1U) + 17U);
  }
}

void validate_append_marker(const std::string& filename, const char* marker, const char* mode)
{
  if (!file_has_content(filename))
    return;
  std::ifstream input(filename);
  std::string first_line;
  if (!std::getline(input, first_line) || first_line != marker)
    mode_error(
      mode,
      "cannot append to " + filename +
        " because its output format is incompatible. Move or remove the existing file "
        "before resuming.");
}

FILE* open_output_file(
  const std::string& filename, const bool resume, const char* marker, bool& fresh)
{
  fresh = !(resume && file_has_content(filename));
  FILE* file = my_fopen(filename.c_str(), resume ? "a" : "w");
  if (fresh)
    fprintf(file, "%s\n", marker);
  return file;
}

void write_xyz_box(FILE* file, const Box& box, const int precision)
{
  fprintf(
    file, "pbc=\"%c %c %c\" ", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');
  fprintf(file, "Lattice=\"");
  const int transpose[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
  for (int i = 0; i < 9; ++i)
    fprintf(file, i == 0 ? "%.*g" : " %.*g", precision, box.cpu_h[transpose[i]]);
  fprintf(file, "\" ");
}

void write_restart_xyz(
  const std::string& filename,
  Replica_Slot& slot,
  const Atom& source_atom,
  const std::string& fields)
{
  CHECK(gpuSetDevice(slot.device_id));
  slot.stream.synchronize();
  slot.atom.position_per_atom.copy_to_host(slot.atom.cpu_position_per_atom.data());
  slot.atom.velocity_per_atom.copy_to_host(slot.atom.cpu_velocity_per_atom.data());

  FILE* file = my_fopen(filename.c_str(), "w");
  const int number_of_atoms = slot.atom.number_of_atoms;
  fprintf(file, "%d\n", number_of_atoms);
  write_xyz_box(file, slot.box, 17);
  fprintf(file, "Properties=%s %s\n", RESTART_XYZ_PROPERTIES, fields.c_str());
  for (int atom = 0; atom < number_of_atoms; ++atom) {
    fprintf(
      file,
      "%s %.17g %.17g %.17g %.17g",
      source_atom.cpu_atom_symbol[atom].c_str(),
      slot.atom.cpu_position_per_atom[atom],
      slot.atom.cpu_position_per_atom[atom + number_of_atoms],
      slot.atom.cpu_position_per_atom[atom + 2 * number_of_atoms],
      source_atom.cpu_mass[atom]);
    write_restart_xyz_velocity(file, slot.atom.cpu_velocity_per_atom, atom);
  }
  fclose(file);
}

std::string read_restart_xyz_header(
  std::ifstream& input,
  const std::string& filename,
  const Atom& source_atom,
  const Box& source_box,
  const char* what)
{
  if (!input.is_open())
    mode_error(what, "restart: cannot open " + filename + ".");
  int number_of_atoms = 0;
  if (!(input >> number_of_atoms) || number_of_atoms != source_atom.number_of_atoms)
    mode_error(what, "restart atom count does not match model.xyz.");
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string comment;
  if (!std::getline(input, comment))
    mode_error(what, "restart xyz has no comment line.");
  validate_xyz_box(comment, source_box, what);
  return comment;
}

void read_restart_xyz_atoms(
  std::istream& input,
  const std::string& comment,
  const Atom& source_atom,
  Atom& atom,
  const char* mode)
{
  std::string properties;
  bool natural_velocity = false;
  if (!extract_token_value(comment, "Properties", properties) ||
      !parse_restart_xyz_properties(properties, natural_velocity))
    mode_error(mode, "restart xyz Properties are invalid.");
  const int n = source_atom.number_of_atoms;
  for (int i = 0; i < n; ++i) {
    std::string symbol;
    double mass = 0.0;
    if (!(input >> symbol >> atom.cpu_position_per_atom[i] >>
          atom.cpu_position_per_atom[i + n] >> atom.cpu_position_per_atom[i + 2 * n] >> mass) ||
        !read_restart_xyz_velocity(input, natural_velocity, atom.cpu_velocity_per_atom, i))
      mode_error(mode, "restart xyz atom data is incomplete.");
    if (symbol != source_atom.cpu_atom_symbol[i] || !values_match(mass, source_atom.cpu_mass[i]))
      mode_error(mode, "restart xyz species or mass does not match model.xyz.");
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(atom.cpu_position_per_atom[i + d * n]) ||
          !std::isfinite(atom.cpu_velocity_per_atom[i + d * n]))
        mode_error(mode, "restart xyz contains a non-finite atom value.");
    }
  }
}

void read_restart_metadata_header(
  std::istream& input, const char* mode, const char* mode_key, const int format_version)
{
  std::string header;
  if (!std::getline(input, header) ||
      header != "# GPUMD " + std::string(mode) + " restart metadata")
    mode_error(mode, "restart metadata has an invalid header.");
  int version = 0;
  read_restart_value(input, mode, "format_version", version);
  if (version != format_version)
    mode_error("Unsupported", std::string(mode) + " restart format_version.");
  std::string stored_mode;
  read_restart_value(input, mode, "mode", stored_mode);
  if (stored_mode != mode_key)
    mode_error(mode, "restart mode is invalid.");
}

void read_restart_key(std::istream& input, const char* mode, const char* expected_key)
{
  std::string key;
  if (!(input >> key) || key != expected_key)
    mode_error(mode, "restart key " + std::string(expected_key) + " is invalid or missing.");
}

std::string read_restart_line(std::istream& input, const char* mode, const char* expected_key)
{
  read_restart_key(input, mode, expected_key);
  std::string value;
  std::getline(input, value);
  value = trim_left(value);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\r'))
    value.pop_back();
  if (value.empty())
    mode_error(mode, "restart value of " + std::string(expected_key) + " is empty.");
  return value;
}

void read_restart_bdp_states(
  std::istream& input, const char* mode, const int replicas, std::vector<std::string>& states)
{
  states.resize(replicas);
  for (int replica = 0; replica < replicas; ++replica) {
    int stored_replica = -1;
    read_restart_key(input, mode, "bdp_rng");
    if (!(input >> stored_replica) || stored_replica != replica)
      mode_error(mode, "restart bdp_rng replica id is invalid.");
    std::getline(input, states[replica]);
    states[replica] = trim_left(states[replica]);
    if (states[replica].empty())
      mode_error(mode, "restart bdp_rng state is empty.");
  }
  std::string extra;
  if (input >> extra)
    mode_error(mode, "restart metadata has unexpected trailing data.");
}

Replica_Slot::Replica_Slot(const int replica_id, const int device_id)
  : replica_id(replica_id)
  , device_id(device_id)
  , stream(device_id)
  , complete(device_id)
  , host_thermo(2, 0.0)
{
}

Replica_Slot::~Replica_Slot()
{
  CHECK(gpuSetDevice(device_id));
  if (stream.valid())
    stream.synchronize();
}

Replica_State::Replica_State(
  const int device_id,
  const int number_of_atoms,
  const bool store_velocity)
  : device_id_(device_id)
  , store_velocity_(store_velocity)
{
  gpu_detail::Device_Guard guard(device_id_);
  position_.resize(static_cast<size_t>(number_of_atoms) * 3);
  if (store_velocity_)
    velocity_.resize(static_cast<size_t>(number_of_atoms) * 3);
}

Replica_State::~Replica_State()
{
  // GPU_Vector releases its storage after this destructor body.
  CHECK(gpuSetDevice(device_id_));
}

void Replica_State::save(Replica_Slot& slot)
{
  if (slot.device_id != device_id_)
    PRINT_INPUT_ERROR("Cannot save a replica state on a different GPU.");
  CHECK(gpuSetDevice(device_id_));
  position_.copy_from_device_async(
    slot.atom.position_per_atom.data(), slot.stream.get());
  if (store_velocity_) {
    velocity_.copy_from_device_async(
      slot.atom.velocity_per_atom.data(), slot.stream.get());
  }
}

void Replica_State::restore(Replica_Slot& slot)
{
  if (slot.device_id != device_id_)
    PRINT_INPUT_ERROR("Cannot restore a replica state on a different GPU.");
  CHECK(gpuSetDevice(device_id_));
  position_.copy_to_device_async(
    slot.atom.position_per_atom.data(), slot.stream.get());
  if (store_velocity_) {
    velocity_.copy_to_device_async(
      slot.atom.velocity_per_atom.data(), slot.stream.get());
  }
}

void Replica_State::copy_position_to_host(
  GPU_Pinned_Vector<double>& host_position, Replica_Slot& slot)
{
  if (
    slot.device_id != device_id_ ||
    host_position.size() != position_.size())
    PRINT_INPUT_ERROR("Invalid host buffer for a replica position copy.");
  CHECK(gpuSetDevice(device_id_));
  position_.copy_to_host_async(host_position.data(), slot.stream.get());
  slot.stream.synchronize();
}

void Replica_State::copy_position_from_host(
  const GPU_Pinned_Vector<double>& host_position, Replica_Slot& slot)
{
  if (
    slot.device_id != device_id_ ||
    host_position.size() != position_.size())
    PRINT_INPUT_ERROR("Invalid host buffer for a replica position copy.");
  CHECK(gpuSetDevice(device_id_));
  position_.copy_from_host_async(host_position.data(), slot.stream.get());
}

Device_Context::Device_Context(const int device_id) : device_id(device_id) {}

Device_Context::~Device_Context()
{
  if (worker.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    condition.notify_one();
    worker.join();
  }
  CHECK(gpuSetDevice(device_id));
  batch.reset();
  slots.clear();
  force.clear();
}

Replica_Runtime::Replica_Runtime(
  const std::vector<std::vector<std::string>>& potential_commands,
  const RunInput& run_input,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group,
  const double time_step)
  : potential_commands_(potential_commands)
  , run_input_(run_input)
  , source_atom_(source_atom)
  , source_box_(source_box)
  , source_group_(source_group)
  , time_step_(time_step)
{
}

Replica_Runtime::~Replica_Runtime()
{
  stop_workers();
  report_batches();
  devices_.clear();
}

void Replica_Runtime::report_batches() const
{
  if (!batch_active_)
    return;
  Replica_Batch_Statistics total;
  for (const auto& device : devices_) {
    if (!device->batch)
      continue;
    const Replica_Batch_Statistics& s = device->batch->statistics();
    total.advances += s.advances;
    total.steps += s.steps;
  }
  printf("Batched integration: %lld advances, %lld steps.\n", total.advances, total.steps);
  fflush(stdout);
}

void Replica_Runtime::clone_source_state(Replica_Slot& slot) const
{
  slot.atom.number_of_atoms = source_atom_.number_of_atoms;
  slot.atom.cpu_type = source_atom_.cpu_type;
  slot.atom.cpu_type_size = source_atom_.cpu_type_size;
  slot.atom.cpu_mass = source_atom_.cpu_mass;
  slot.atom.cpu_charge = source_atom_.cpu_charge;
  slot.atom.cpu_position_per_atom = source_atom_.cpu_position_per_atom;
  slot.atom.cpu_velocity_per_atom = source_atom_.cpu_velocity_per_atom;
  slot.atom.cpu_atom_symbol = source_atom_.cpu_atom_symbol;
  slot.box = source_box_;
  slot.group.resize(source_group_.size());
  for (size_t method = 0; method < source_group_.size(); ++method) {
    slot.group[method].number = source_group_[method].number;
    slot.group[method].cpu_label = source_group_[method].cpu_label;
    slot.group[method].cpu_size = source_group_[method].cpu_size;
    slot.group[method].cpu_size_sum = source_group_[method].cpu_size_sum;
    slot.group[method].cpu_contents = source_group_[method].cpu_contents;
  }
}

void Replica_Runtime::initialize_device_force(Device_Context& device)
{
  CHECK(gpuSetDevice(device.device_id));
  if (potential_commands_[0].size() != 2)
    PRINT_INPUT_ERROR("potential should have one parameter.");
  device.force.initialize(
    potential_commands_[0][1],
    source_box_,
    source_atom_.number_of_atoms,
    run_input_,
    devices_.empty() || &device == devices_.front().get());
}

void Replica_Runtime::compute_force(Device_Context& device, Replica_Slot& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  if (!slot.thermostat)
    PRINT_INPUT_ERROR("Replica slot has no thermostat.");
  device.force.prepare_stream(slot.stream.get());
  device.force.compute(
    slot.box,
    slot.atom.position_per_atom,
    slot.atom.type,
    slot.atom.potential_per_atom,
    slot.atom.force_per_atom,
    slot.atom.virial_per_atom,
    slot.stream.get());
  slot.thermostat->find_thermo(slot.box.get_volume(), slot.atom, slot.thermo, slot.stream.get());
  slot.thermo.copy_to_host_async(slot.host_thermo.data(), 2, slot.stream.get());
  slot.complete.record(slot.stream);
  slot.complete.synchronize();
  slot.potential_energy = slot.host_thermo[1];
}

Replica_Force& Replica_Runtime::force(Replica_Slot& slot)
{
  for (auto& device : devices_) {
    if (device->device_id == slot.device_id)
      return device->force;
  }
  PRINT_INPUT_ERROR("Replica slot is not assigned to an active GPU.");
  return devices_[0]->force;
}

void Replica_Runtime::initialize(
  const int replicas,
  const int replicas_per_gpu,
  const Slot_Initializer& initialize_slot)
{
  if (!devices_.empty() || !replica_slots_.empty())
    PRINT_INPUT_ERROR("Replica runtime cannot be initialized more than once.");
  if (replicas <= 0 || replicas_per_gpu <= 0)
    PRINT_INPUT_ERROR("Replica counts should be positive.");

  int number_of_devices = 0;
  CHECK(gpuGetDeviceCount(&number_of_devices));
  if (number_of_devices <= 0)
    PRINT_INPUT_ERROR("No GPU device is available for multi_replica.");
  const int used_devices = std::min(number_of_devices, replicas);
  if (replicas_per_gpu * used_devices < replicas)
    PRINT_INPUT_ERROR("replicas_per_gpu is too small for the visible GPUs.");

  devices_.reserve(used_devices);
  for (int device_id = 0; device_id < used_devices; ++device_id) {
    std::unique_ptr<Device_Context> device(new Device_Context(device_id));
    initialize_device_force(*device);
    devices_.push_back(std::move(device));
  }

  replica_slots_.assign(replicas, nullptr);
  for (int replica_id = 0; replica_id < replicas; ++replica_id) {
    Device_Context& device = *devices_[replica_id % used_devices];
    if (static_cast<int>(device.slots.size()) >= replicas_per_gpu)
      PRINT_INPUT_ERROR("Internal slot assignment exceeded replicas_per_gpu.");
    CHECK(gpuSetDevice(device.device_id));
    std::unique_ptr<Replica_Slot> slot(new Replica_Slot(replica_id, device.device_id));
    clone_source_state(*slot);
    initialize_slot(*slot);
    compute_force(device, *slot);
    replica_slots_[replica_id] = slot.get();
    device.slots.push_back(std::move(slot));
  }

  initialize_batches();

  printf("    visible/used GPUs: %d/%d; slot distribution:", number_of_devices, used_devices);
  for (const auto& device : devices_) {
    device->active_slots.resize(device->slots.size(), 1);
    printf(" GPU%d=%zu", device->device_id, device->slots.size());
  }
  printf(".\n");
  fflush(stdout);
}

// Batched integration pays off while the per-replica path is limited by kernel launches. Measured
// on an RTX 4090 (Cu, NEP, 55 to 32000 atoms per replica, 2 to 64 replicas), it is faster for any
// number of replicas up to about 5000 atoms per replica, and for larger replicas while the GPU holds
// at most about 30000 atoms in total; otherwise it is up to 20% slower. GPUMD_REPLICA_BATCH=0 or 1
// overrides the choice, e.g. for tests.
constexpr long long max_batched_atoms_per_replica = 5000;
constexpr long long max_batched_atoms_per_gpu = 30000;

void Replica_Runtime::initialize_batches()
{
  const char* forced = std::getenv("GPUMD_REPLICA_BATCH");
  const bool is_forced = forced != nullptr && forced[0] != '\0';
  if (is_forced && std::string(forced) != "0" && std::string(forced) != "1")
    PRINT_INPUT_ERROR("GPUMD_REPLICA_BATCH should be 0 or 1.");
  if (is_forced && forced[0] == '0') {
    printf("    batched integration: off (GPUMD_REPLICA_BATCH=0).\n");
    return;
  }
  if (!is_forced) {
    size_t slots_per_device = 0;
    for (const auto& device : devices_)
      slots_per_device = std::max(slots_per_device, device->slots.size());
    const long long atoms_per_device =
      static_cast<long long>(slots_per_device) * source_atom_.number_of_atoms;
    if (slots_per_device < 2) {
      printf("    batched integration: off (one replica per GPU).\n");
      return;
    }
    const long long atoms_per_replica = source_atom_.number_of_atoms;
    if (
      atoms_per_replica > max_batched_atoms_per_replica &&
      atoms_per_device > max_batched_atoms_per_gpu) {
      printf(
        "    batched integration: off (%lld atoms per replica > %lld and %lld atoms per GPU > "
        "%lld).\n",
        atoms_per_replica,
        max_batched_atoms_per_replica,
        atoms_per_device,
        max_batched_atoms_per_gpu);
      return;
    }
  }
  const std::string& potential_file = potential_commands_[0][1];
  if (devices_.front()->force.is_charge_nep()) {
    printf("    batched integration: off (not supported for qNEP).\n");
    return;
  }
  for (auto& device : devices_) {
    CHECK(gpuSetDevice(device->device_id));
    std::unique_ptr<Replica_Batch> batch(new Replica_Batch(
      device->device_id,
      static_cast<int>(device->slots.size()),
      potential_file,
      source_atom_,
      source_box_));
    if (!batch->supported()) {
      for (auto& other : devices_)
        other->batch.reset();
      printf("    batched integration: off (the box needs the small-box NEP path).\n");
      return;
    }
    device->batch = std::move(batch);
  }
  batch_active_ = true;
  printf("    batched integration: on (one concatenated system per GPU).\n");
}

void Replica_Runtime::start_workers()
{
  for (auto& device : devices_) {
    if (!device->worker.joinable()) {
      device->worker =
        std::thread(&Replica_Runtime::worker_loop, this, std::ref(*device));
    }
  }
}

void Replica_Runtime::stop_workers()
{
  for (auto& device : devices_) {
    if (!device->worker.joinable())
      continue;
    {
      std::lock_guard<std::mutex> lock(device->mutex);
      device->stop = true;
    }
    device->condition.notify_one();
  }
  for (auto& device : devices_) {
    if (device->worker.joinable())
      device->worker.join();
  }
}

void Replica_Runtime::worker_loop(Device_Context& device)
{
  CHECK(gpuSetDevice(device.device_id));
  while (true) {
    std::unique_lock<std::mutex> lock(device.mutex);
    device.condition.wait(lock, [&device] { return device.ready || device.stop; });
    if (device.stop)
      break;
    const int steps = device.steps;
    const std::vector<unsigned char> active_slots = device.active_slots;
    device.ready = false;
    lock.unlock();

    advance_device(device, active_slots, steps);

    lock.lock();
    device.finished = true;
    lock.unlock();
    device.condition.notify_one();
  }
}

void Replica_Runtime::advance_device(
  Device_Context& device,
  const std::vector<unsigned char>& active_slots,
  const int steps)
{
  CHECK(gpuSetDevice(device.device_id));
  if (active_slots.size() != device.slots.size())
    PRINT_INPUT_ERROR("Internal active replica selection is inconsistent.");

  if (device.batch) {
    std::vector<Replica_Slot*> batch_slots;
    batch_slots.reserve(device.slots.size());
    for (size_t local_slot = 0; local_slot < device.slots.size(); ++local_slot) {
      if (active_slots[local_slot])
        batch_slots.push_back(device.slots[local_slot].get());
    }
    if (batch_slots.empty())
      return;
    device.batch->advance(batch_slots, steps, time_step_);
    return;
  }

  for (int local_step = 0; local_step < steps; ++local_step) {
    for (size_t local_slot = 0; local_slot < device.slots.size(); ++local_slot) {
      if (!active_slots[local_slot])
        continue;
      Replica_Slot& slot = *device.slots[local_slot];
      slot.thermostat->compute1(time_step_, slot.atom, slot.stream.get());
      device.force.compute(
        slot.box,
        slot.atom.position_per_atom,
        slot.atom.type,
        slot.atom.potential_per_atom,
        slot.atom.force_per_atom,
        slot.atom.virial_per_atom,
        slot.stream.get());
      slot.thermostat->compute2(
        time_step_, slot.box.get_volume(), slot.atom, slot.thermo, slot.stream.get());
    }
  }

  for (size_t local_slot = 0; local_slot < device.slots.size(); ++local_slot) {
    if (!active_slots[local_slot])
      continue;
    Replica_Slot& slot = *device.slots[local_slot];
    slot.thermo.copy_to_host_async(slot.host_thermo.data(), 2, slot.stream.get());
    slot.complete.record(slot.stream);
  }
  for (size_t local_slot = 0; local_slot < device.slots.size(); ++local_slot) {
    if (!active_slots[local_slot])
      continue;
    Replica_Slot& slot = *device.slots[local_slot];
    slot.complete.synchronize();
    slot.potential_energy = slot.host_thermo[1];
  }
}

void Replica_Runtime::advance(
  const std::vector<unsigned char>& active_replicas,
  const int steps)
{
  if (active_replicas.size() != replica_slots_.size())
    PRINT_INPUT_ERROR("Internal active replica selection has the wrong size.");
  if (steps < 0)
    PRINT_INPUT_ERROR("Replica MD block length cannot be negative.");
  if (steps == 0)
    return;

  for (auto& device : devices_) {
    std::lock_guard<std::mutex> lock(device->mutex);
    if (!device->worker.joinable())
      PRINT_INPUT_ERROR("Replica GPU workers have not been started.");
    device->active_slots.resize(device->slots.size());
    for (size_t local_slot = 0; local_slot < device->slots.size(); ++local_slot) {
      const int replica_id = device->slots[local_slot]->replica_id;
      device->active_slots[local_slot] = active_replicas[replica_id];
    }
    device->steps = steps;
    device->finished = false;
    device->ready = true;
  }
  for (auto& device : devices_)
    device->condition.notify_one();
  for (auto& device : devices_) {
    std::unique_lock<std::mutex> lock(device->mutex);
    device->condition.wait(lock, [&device] { return device->finished; });
  }
  for (size_t replica_id = 0; replica_id < replica_slots_.size(); ++replica_id) {
    if (!active_replicas[replica_id])
      continue;
    const Replica_Slot& slot = *replica_slots_[replica_id];
    if (!std::isfinite(slot.potential_energy) || !std::isfinite(slot.host_thermo[0]))
      PRINT_INPUT_ERROR("multi_replica encountered a non-finite thermodynamic value.");
  }
}

void Replica_Runtime::advance_all(const int steps)
{
  advance(std::vector<unsigned char>(replica_slots_.size(), 1), steps);
}

void Replica_Runtime::compute_forces(
  const std::vector<unsigned char>& active_replicas)
{
  if (active_replicas.size() != replica_slots_.size())
    PRINT_INPUT_ERROR("Internal active replica selection has the wrong size.");

  for (auto& device : devices_) {
    CHECK(gpuSetDevice(device->device_id));
    for (auto& owned_slot : device->slots) {
      Replica_Slot& slot = *owned_slot;
      if (!active_replicas[slot.replica_id])
        continue;
      device->force.compute(
        slot.box,
        slot.atom.position_per_atom,
        slot.atom.type,
        slot.atom.potential_per_atom,
        slot.atom.force_per_atom,
        slot.atom.virial_per_atom,
        slot.stream.get());
      slot.complete.record(slot.stream);
    }
  }
  for (auto& device : devices_) {
    for (auto& owned_slot : device->slots) {
      Replica_Slot& slot = *owned_slot;
      if (active_replicas[slot.replica_id])
        slot.complete.synchronize();
    }
  }
}

void Replica_Runtime::compute_forces_all()
{
  compute_forces(std::vector<unsigned char>(replica_slots_.size(), 1));
}

void Replica_Runtime::compute_forces_selected(
  const std::vector<int>& replica_ids)
{
  std::vector<unsigned char> active_replicas(replica_slots_.size(), 0);
  for (const int replica_id : replica_ids) {
    if (replica_id < 0 || replica_id >= static_cast<int>(replica_slots_.size()))
      PRINT_INPUT_ERROR("Selected replica ID is out of range.");
    active_replicas[replica_id] = 1;
  }
  compute_forces(active_replicas);
}

void Replica_Runtime::advance_selected(
  const std::vector<int>& replica_ids,
  const int steps)
{
  std::vector<unsigned char> active_replicas(replica_slots_.size(), 0);
  for (const int replica_id : replica_ids) {
    if (replica_id < 0 || replica_id >= static_cast<int>(replica_slots_.size()))
      PRINT_INPUT_ERROR("Selected replica ID is out of range.");
    active_replicas[replica_id] = 1;
  }
  advance(active_replicas, steps);
}

} // namespace replica
