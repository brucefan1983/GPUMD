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
Two-dimensional temperature-pressure replica-exchange molecular dynamics in the
isothermal-isobaric ensemble:
[1] T. Okabe, M. Kawata, Y. Okamoto, and M. Mikami,
    Chem. Phys. Lett. 335, 435 (2001).
[2] K. Nomura et al., Proc. Natl. Acad. Sci. USA 114, 4066 (2017).

The REMD-TP feature was proposed and primarily implemented by Jiuyang Shi.
Wenhao Luo ported the implementation into the gpumd multi_replica runtime.

M = M_T x M_P replicas sit on a T-P grid, label = p_index * M_T + t_index. The
grid may be sheared: each temperature row can have its own pressures
(pressure_grid), e.g. centred on an estimated Widom/coexistence line. The
acceptance rule below is exact for any pair of (T, P) conditions.
Each replica runs NPT dynamics with the BDP thermostat and the stochastic cell
rescaling barostat. Exchange rounds cycle through neighbouring pairs along T
(same P) and along P (same T), even pairs first, then odd pairs. A swap of the
conditions of replicas i and j (at labels a and b) is accepted with
  min{1, exp[(beta_a - beta_b)(E_i - E_j) + (beta_a P_a - beta_b P_b)(V_i - V_j)]}.
------------------------------------------------------------------------------*/

#include "replica.cuh"
#include "replica_common.cuh"
#include "utilities/run_input.cuh"
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
#include <limits>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
using replica::Replica_Runtime;
using replica::Replica_Slot;
using replica::SCR_Barostat;
using replica::generate_canonical_velocities;
using replica::extract_quoted_value;
using replica::extract_token_value;
using replica::file_has_content;
using replica::parse_scalar;
using replica::trim_left;
using replica::values_match;

constexpr int restart_format_version = 3;
constexpr const char* exchange_output_marker = "# GPUMD REMD-TP exchange output format 1";
constexpr const char* status_output_marker = "# GPUMD REMD-TP status output format 1";
constexpr const char* thermo_output_marker = "# GPUMD REMD-TP thermodynamic output format 1";

enum Direction { along_temperature = 0, along_pressure = 1 };

bool is_finite_real(const char* text, double* value)
{
  return is_valid_real(text, value) && std::isfinite(*value);
}

void validate_barostat_box(const Box& box)
{
  const double* h = box.cpu_h;
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(h[i]) || (i % 4 == 0 ? h[i] <= 0.0 : h[i] != 0.0))
      PRINT_INPUT_ERROR("REMD-TP requires an orthogonal box with positive finite lengths.");
  }
  if (!std::isfinite(box.get_volume()) || box.get_volume() <= 0.0)
    PRINT_INPUT_ERROR("REMD-TP requires a positive finite box volume.");
}

int next_periodic_event(
  const int local_step, const int final_local_step, const long long global_step, const int interval)
{
  const long long remainder = global_step % interval;
  const long long steps = remainder == 0 ? interval : interval - remainder;
  return static_cast<int>(
    std::min<long long>(final_local_step, static_cast<long long>(local_step) + steps));
}

void read_value_file(const std::string& filename, std::vector<double>& values, const char* what)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    const std::string message = std::string("Cannot open the REMD-TP ") + what + " file.";
    PRINT_INPUT_ERROR(message.c_str());
  }
  while (input.peek() != EOF) {
    const std::vector<std::string> tokens = get_tokens(input);
    for (const std::string& token : tokens) {
      if (token.empty() || token[0] == '#')
        break;
      double value = 0.0;
      if (!is_finite_real(token.c_str(), &value)) {
        const std::string message = std::string("Invalid value in the REMD-TP ") + what + " file.";
        PRINT_INPUT_ERROR(message.c_str());
      }
      values.push_back(value);
    }
  }
}

// Parses "<min> <max> <count>" or "<file>" starting at command[i].
void parse_ladder(
  const std::vector<std::string>& command,
  size_t& i,
  std::vector<double>& values,
  double& minimum,
  double& maximum,
  int& count,
  bool& from_range,
  const char* what)
{
  const std::string name(what);
  if (++i >= command.size())
    PRINT_INPUT_ERROR((name + " requires a file name or <min> <max> <count>.").c_str());
  if (is_finite_real(command[i].c_str(), &minimum)) {
    from_range = true;
    if (i + 2 >= command.size() || !is_finite_real(command[i + 1].c_str(), &maximum) ||
        !is_valid_int(command[i + 2].c_str(), &count) || count <= 0)
      PRINT_INPUT_ERROR((name + " should be followed by <min> <max> <count>.").c_str());
    if ((count == 1 && !values_match(minimum, maximum)) || (count > 1 && !(maximum > minimum)))
      PRINT_INPUT_ERROR(
        (name + " needs min < max, or min = max with a count of 1.").c_str());
    i += 3;
  } else {
    from_range = false;
    read_value_file(command[i++], values, what);
    count = static_cast<int>(values.size());
    if (count == 0)
      PRINT_INPUT_ERROR((name + " file is empty.").c_str());
  }
}

struct REMD_TP_Config {
  enum Spacing { geometric, linear };

  int replicas = 0;
  int replicas_per_gpu = 0;
  int exchange_interval = 0;
  int equilibration_steps = 0;
  int dump_interval = 0;
  int num_temperatures = 0;
  int num_pressures = 0;
  bool resume = false;
  bool dump_xyz = false;
  bool dump_velocity = false;
  std::string resume_prefix;
  std::string dump_prefix;
  std::vector<double> temperatures; // K, length num_temperatures
  std::vector<double> pressures;    // GPa, rectangular ladder (unused with pressure_grid)
  // GPa, num_temperatures x num_pressures, row t holds the pressures of temperature t.
  // Rectangular grids repeat the same row; pressure_grid allows a sheared grid.
  std::vector<double> pressure_table;
  unsigned int internal_seed = 1;

  void parse(const std::vector<std::string>& command, const std::vector<std::string>& dump_command);
};

void REMD_TP_Config::parse(
  const std::vector<std::string>& command,
  const std::vector<std::string>& dump_command)
{
  if (command.size() < 3 || command[0] != "multi_replica" || command[1] != "remd_tp")
    PRINT_INPUT_ERROR("Expected multi_replica remd_tp followed by its options.");

  Spacing temperature_spacing = geometric;
  bool has_temperature = false;
  bool has_pressure = false;
  bool temperature_from_range = false;
  bool pressure_from_range = false;
  bool pressure_from_grid = false;
  std::string pressure_grid_file;
  double t_min = 0.0, t_max = 0.0, p_min = 0.0, p_max = 0.0;
  for (size_t i = 2; i < command.size();) {
    if (command[i] == "replicas") {
      if (++i >= command.size() || !is_valid_int(command[i].c_str(), &replicas) || replicas <= 1)
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
      if (++i >= command.size() || !is_valid_int(command[i].c_str(), &exchange_interval) ||
          exchange_interval <= 0)
        PRINT_INPUT_ERROR("exchange should be a positive integer.");
      ++i;
    } else if (command[i] == "equilibrate") {
      if (++i >= command.size() || !is_valid_int(command[i].c_str(), &equilibration_steps) ||
          equilibration_steps < 0)
        PRINT_INPUT_ERROR("equilibrate should be a non-negative integer.");
      ++i;
    } else if (command[i] == "temp") {
      if (has_temperature)
        PRINT_INPUT_ERROR("Specify the REMD-TP temperatures only once.");
      parse_ladder(
        command, i, temperatures, t_min, t_max, num_temperatures, temperature_from_range, "temp");
      has_temperature = true;
    } else if (command[i] == "pressure") {
      if (has_pressure)
        PRINT_INPUT_ERROR("Specify the REMD-TP pressures only once.");
      parse_ladder(
        command, i, pressures, p_min, p_max, num_pressures, pressure_from_range, "pressure");
      has_pressure = true;
    } else if (command[i] == "pressure_grid") {
      if (has_pressure)
        PRINT_INPUT_ERROR("Specify the REMD-TP pressures only once.");
      if (++i >= command.size())
        PRINT_INPUT_ERROR("pressure_grid requires a file name.");
      pressure_grid_file = command[i++];
      pressure_from_grid = true;
      has_pressure = true;
    } else if (command[i] == "spacing") {
      if (++i >= command.size())
        PRINT_INPUT_ERROR("spacing requires geometric or linear.");
      if (command[i] == "geometric")
        temperature_spacing = geometric;
      else if (command[i] == "linear")
        temperature_spacing = linear;
      else
        PRINT_INPUT_ERROR("spacing should be geometric or linear.");
      ++i;
    } else if (command[i] == "resume") {
      if (++i >= command.size())
        PRINT_INPUT_ERROR("resume requires a restart prefix.");
      resume = true;
      resume_prefix = command[i++];
    } else {
      PRINT_INPUT_ERROR("Unknown multi_replica remd_tp option.");
    }
  }

  if (replicas <= 1 || exchange_interval <= 0 || !has_temperature || !has_pressure)
    PRINT_INPUT_ERROR("REMD-TP requires replicas, exchange, temp, and pressure options.");
  if (resume && equilibration_steps > 0)
    PRINT_INPUT_ERROR("REMD-TP resume cannot be combined with equilibrate.");
  if (num_temperatures > replicas || (!pressure_from_grid && num_pressures > replicas))
    PRINT_INPUT_ERROR("REMD-TP grid dimensions cannot exceed the replica count.");
  if (temperature_from_range) {
    temperatures.resize(num_temperatures);
    for (int t = 0; t < num_temperatures; ++t) {
      const double f = num_temperatures == 1 ? 0.0 : static_cast<double>(t) / (num_temperatures - 1);
      temperatures[t] = temperature_spacing == linear ? t_min + f * (t_max - t_min)
                                                      : t_min * std::pow(t_max / t_min, f);
    }
  }
  if (pressure_from_range) {
    pressures.resize(num_pressures);
    for (int p = 0; p < num_pressures; ++p) {
      const double f = num_pressures == 1 ? 0.0 : static_cast<double>(p) / (num_pressures - 1);
      pressures[p] = p_min + f * (p_max - p_min);
    }
  }
  for (int t = 0; t < num_temperatures; ++t) {
    if (!std::isfinite(temperatures[t]) || !(temperatures[t] > 0.0) || (t > 0 && !(temperatures[t] > temperatures[t - 1])))
      PRINT_INPUT_ERROR("REMD-TP temperatures should be positive and strictly increasing.");
  }
  if (pressure_from_grid) {
    // One line per temperature (in increasing temperature order), M_P pressures per line.
    std::ifstream input(pressure_grid_file);
    if (!input.is_open())
      PRINT_INPUT_ERROR("Cannot open the REMD-TP pressure_grid file.");
    int rows = 0;
    std::string line;
    while (std::getline(input, line)) {
      const size_t comment = line.find('#');
      if (comment != std::string::npos)
        line = line.substr(0, comment);
      std::istringstream row(line);
      std::vector<double> values;
      std::string token;
      while (row >> token) {
        double value = 0.0;
        if (!is_finite_real(token.c_str(), &value))
          PRINT_INPUT_ERROR("Invalid value in the REMD-TP pressure_grid file.");
        values.push_back(value);
      }
      if (values.empty())
        continue;
      if (rows >= num_temperatures || values.size() > static_cast<size_t>(replicas))
        PRINT_INPUT_ERROR("REMD-TP pressure_grid dimensions exceed the replica count.");
      if (rows == 0)
        num_pressures = static_cast<int>(values.size());
      else if (static_cast<int>(values.size()) != num_pressures)
        PRINT_INPUT_ERROR("Every pressure_grid row should have the same number of pressures.");
      pressure_table.insert(pressure_table.end(), values.begin(), values.end());
      ++rows;
    }
    if (rows != num_temperatures)
      PRINT_INPUT_ERROR("pressure_grid should have one row per REMD-TP temperature.");
  } else {
    for (int t = 0; t < num_temperatures; ++t)
      pressure_table.insert(pressure_table.end(), pressures.begin(), pressures.end());
  }
  for (int t = 0; t < num_temperatures; ++t) {
    for (int p = 1; p < num_pressures; ++p) {
      if (!(pressure_table[t * num_pressures + p] > pressure_table[t * num_pressures + p - 1]))
        PRINT_INPUT_ERROR("REMD-TP pressures should be strictly increasing within each temperature.");
    }
  }
  if (static_cast<long long>(num_temperatures) * num_pressures != replicas)
    PRINT_INPUT_ERROR("REMD-TP replicas should equal (number of temperatures) x (number of pressures).");

  if (!dump_command.empty()) {
    if (dump_command.size() < 5 || dump_command[0] != "dump_xyz")
      PRINT_INPUT_ERROR("Invalid REMD-TP dump_xyz command.");
    int grouping_method = 0;
    int group_id = 0;
    if (!is_valid_int(dump_command[1].c_str(), &grouping_method) || grouping_method != -1 ||
        !is_valid_int(dump_command[2].c_str(), &group_id))
      PRINT_INPUT_ERROR("REMD-TP dump_xyz requires grouping_method -1.");
    if (!is_valid_int(dump_command[3].c_str(), &dump_interval) || dump_interval <= 0)
      PRINT_INPUT_ERROR("REMD-TP dump_xyz interval should be positive.");
    dump_prefix = dump_command[4];
    if (dump_prefix.empty() || dump_prefix.back() == '*')
      PRINT_INPUT_ERROR("REMD-TP dump_xyz does not support an empty or '*' filename.");
    for (size_t i = 5; i < dump_command.size(); ++i) {
      if (dump_command[i] == "velocity")
        dump_velocity = true;
      else
        PRINT_INPUT_ERROR("REMD-TP dump_xyz supports positions and optional velocity only.");
    }
    dump_xyz = true;
  }

  internal_seed = replica::draw_internal_seed();
}

// ensemble npt_scr <T_1> <T_2> <tau_T> <P_1> <P_2> <bulk_modulus> <tau_p>
//                  [couple <xyz|xy|xz|yz|x|y|z>] [pcouple_interval <steps>]
struct Barostat_Input {
  double temperature_coupling = 0.0;
  double bulk_modulus = 0.0; // GPa
  double pressure_coupling = 0.0;
  int interval = 10;
  int couple[3] = {1, 1, 1};
  std::string couple_name = "xyz";
};

std::string trajectory_filename(const std::string& prefix, const int replica)
{
  const std::string suffix = "_replica_" + std::to_string(replica);
  const size_t separator = prefix.find_last_of("/\\");
  const size_t extension = prefix.find_last_of('.');
  if (extension != std::string::npos && (separator == std::string::npos || extension > separator))
    return prefix.substr(0, extension) + suffix + prefix.substr(extension);
  return prefix + suffix + ".xyz";
}

class REMD_TP_Driver
{
public:
  REMD_TP_Driver(
    const replica::Replica_Input& input, const RunInput& run_input,
    Atom& source_atom, Box& source_box, std::vector<Group>& source_group);
  ~REMD_TP_Driver();
  void run();

private:
  REMD_TP_Config config_;
  Barostat_Input barostat_input_;
  const std::vector<std::vector<std::string>>& potential_commands_;
  const std::vector<std::string>& ensemble_command_;
  Atom& source_atom_;
  Box& source_box_;
  double time_step_;
  int number_of_steps_;
  unsigned long long potential_hash_ = 0;
  long long completed_steps_ = 0;
  long long exchange_count_ = 0;
  std::mt19937 exchange_rng_;
  std::vector<std::pair<Direction, int>> phases_;
  std::vector<std::string> restart_bdp_states_;
  std::vector<std::string> restart_scr_states_;
  std::vector<int> replica_to_label_;
  std::vector<int> label_to_replica_;
  // Pair statistics, indexed by the lower label of a neighbouring pair.
  std::vector<long long> attempts_[2];
  std::vector<long long> accepts_[2];
  std::vector<double> probability_sum_[2];
  Replica_Runtime runtime_;
  std::vector<FILE*> trajectory_files_;

  int num_t() const { return config_.num_temperatures; }
  int num_p() const { return config_.num_pressures; }
  int label_of(const int t, const int p) const { return p * num_t() + t; }
  double temperature_of(const int label) const { return config_.temperatures[label % num_t()]; }
  double pressure_of(const int label) const
  {
    return config_.pressure_table[(label % num_t()) * num_p() + label / num_t()];
  }
  double min_pressure() const
  {
    return *std::min_element(config_.pressure_table.begin(), config_.pressure_table.end());
  }
  double max_pressure() const
  {
    return *std::max_element(config_.pressure_table.begin(), config_.pressure_table.end());
  }
  double beta_of(const int label) const { return 1.0 / (K_B * temperature_of(label)); }

  void parse_ensemble();
  void validate_input();
  void build_phases();
  void initialize_slot(Replica_Slot& slot);
  void set_condition(Replica_Slot& slot, int label);
  void equilibrate();
  void attempt_pair(long long step, Direction direction, int label_a, int label_b, FILE* file);
  void attempt_exchanges(long long step, FILE* file);
  void scale_velocity(Replica_Slot& slot, double factor);
  void write_conditions() const;
  void write_status(long long step, FILE* file) const;
  void write_thermo(long long step, FILE* file) const;
  void write_pair_report() const;
  void open_trajectory_files();
  void close_trajectory_files();
  void write_xyz_header(FILE* file, const Replica_Slot& slot, long long step, bool restart) const;
  void write_trajectories(long long step);
  void write_restart();
  void read_restart_metadata();
  void read_restart_xyz(Replica_Slot& slot);
};

REMD_TP_Driver::REMD_TP_Driver(
  const replica::Replica_Input& input, const RunInput& run_input,
  Atom& source_atom, Box& source_box, std::vector<Group>& source_group)
  : potential_commands_(input.potential_commands)
  , ensemble_command_(input.ensemble_command)
  , source_atom_(source_atom)
  , source_box_(source_box)
  , time_step_(input.time_step)
  , number_of_steps_(input.number_of_steps)
  , runtime_(input.model_commands, run_input, source_atom, source_box, source_group, input.time_step)
{
  config_.parse(input.multi_replica_command, input.dump_xyz_command);
}

REMD_TP_Driver::~REMD_TP_Driver()
{
  runtime_.stop_workers();
  close_trajectory_files();
}

void REMD_TP_Driver::parse_ensemble()
{
  const auto& e = ensemble_command_;
  if (e.size() < 9 || e[0] != "ensemble" || e[1] != "npt_scr")
    PRINT_INPUT_ERROR(
      "REMD-TP requires: ensemble npt_scr <T_1> <T_2> <tau_T> <P_1> <P_2> <bulk_modulus> <tau_p> "
      "[couple <dirs>] [pcouple_interval <steps>].");
  double t1 = 0.0, t2 = 0.0, p1 = 0.0, p2 = 0.0;
  if (!is_finite_real(e[2].c_str(), &t1) || !is_finite_real(e[3].c_str(), &t2) ||
      !is_finite_real(e[4].c_str(), &barostat_input_.temperature_coupling) ||
      barostat_input_.temperature_coupling < 1.0)
    PRINT_INPUT_ERROR("Invalid npt_scr temperature parameters for REMD-TP.");
  if (!is_finite_real(e[5].c_str(), &p1) || !is_finite_real(e[6].c_str(), &p2))
    PRINT_INPUT_ERROR("Invalid npt_scr pressures for REMD-TP.");
  if (!is_finite_real(e[7].c_str(), &barostat_input_.bulk_modulus) ||
      barostat_input_.bulk_modulus <= 0.0)
    PRINT_INPUT_ERROR("npt_scr bulk modulus should be positive (GPa).");
  if (!is_finite_real(e[8].c_str(), &barostat_input_.pressure_coupling) ||
      barostat_input_.pressure_coupling < 1.0)
    PRINT_INPUT_ERROR("npt_scr tau_p should be at least 1 step.");
  for (size_t i = 9; i < e.size();) {
    if (e[i] == "couple" && i + 1 < e.size()) {
      const std::string& dirs = e[i + 1];
      for (int d = 0; d < 3; ++d)
        barostat_input_.couple[d] = 0;
      for (const char c : dirs) {
        if (c < 'x' || c > 'z' || barostat_input_.couple[c - 'x'])
          PRINT_INPUT_ERROR("couple should be a combination of x, y, z without repeats.");
        barostat_input_.couple[c - 'x'] = 1;
      }
      if (dirs.empty())
        PRINT_INPUT_ERROR("couple needs at least one direction.");
      barostat_input_.couple_name = dirs;
      i += 2;
    } else if (e[i] == "pcouple_interval" && i + 1 < e.size()) {
      if (!is_valid_int(e[i + 1].c_str(), &barostat_input_.interval) ||
          barostat_input_.interval <= 0)
        PRINT_INPUT_ERROR("pcouple_interval should be a positive integer.");
      i += 2;
    } else {
      PRINT_INPUT_ERROR("Unknown npt_scr option for REMD-TP.");
    }
  }
  if (!values_match(t1, config_.temperatures.front()) ||
      !values_match(t2, config_.temperatures.back()))
    PRINT_INPUT_ERROR("npt_scr temperatures should match the first and last REMD-TP temperatures.");
  if (!values_match(p1, min_pressure()) || !values_match(p2, max_pressure()))
    PRINT_INPUT_ERROR("npt_scr pressures should match the lowest and highest REMD-TP pressures.");
  if (barostat_input_.pressure_coupling < 2.0 * barostat_input_.interval)
    PRINT_INPUT_ERROR("npt_scr tau_p should be at least twice pcouple_interval.");
}

void REMD_TP_Driver::validate_input()
{
  if (number_of_steps_ <= 0)
    PRINT_INPUT_ERROR("REMD-TP run steps should be positive.");
  if (source_atom_.number_of_atoms < 2)
    PRINT_INPUT_ERROR("REMD-TP requires at least two atoms.");
  std::string kspace_method = "pppm";
  potential_hash_ = replica::validate_potential(
    potential_commands_, source_box_, "REMD-TP", kspace_method);
  if (kspace_method != "none")
    PRINT_INPUT_ERROR("REMD-TP currently supports energy NEP models only (no qNEP).");
  parse_ensemble();

  validate_barostat_box(source_box_);
  const int pbc[3] = {source_box_.pbc_x, source_box_.pbc_y, source_box_.pbc_z};
  for (int d = 0; d < 3; ++d) {
    if (barostat_input_.couple[d] && !pbc[d])
      PRINT_INPUT_ERROR("REMD-TP barostat directions must be periodic.");
  }

  config_.replicas_per_gpu = replica::resolve_replicas_per_gpu(
    config_.replicas, config_.replicas_per_gpu, "REMD-TP");

  replica_to_label_.resize(config_.replicas);
  label_to_replica_.resize(config_.replicas);
  for (int r = 0; r < config_.replicas; ++r)
    replica_to_label_[r] = label_to_replica_[r] = r;
  for (int d = 0; d < 2; ++d) {
    attempts_[d].assign(config_.replicas, 0);
    accepts_[d].assign(config_.replicas, 0);
    probability_sum_[d].assign(config_.replicas, 0.0);
  }
  restart_bdp_states_.resize(config_.replicas);
  restart_scr_states_.resize(config_.replicas);
  build_phases();

  if (config_.resume) {
    read_restart_metadata();
    replica::validate_append_marker("remd_tp_exchange.out", exchange_output_marker, "REMD-TP");
    replica::validate_append_marker("remd_tp_status.out", status_output_marker, "REMD-TP");
    replica::validate_append_marker("remd_tp_thermo.out", thermo_output_marker, "REMD-TP");
  } else {
    exchange_rng_.seed(config_.internal_seed);
  }

  printf("Temperature-pressure replica-exchange molecular dynamics (NPT).\n");
  printf("    replicas: %d = %d temperatures x %d pressures.\n",
         config_.replicas, num_t(), num_p());
  printf("    replicas per GPU: %d.\n", config_.replicas_per_gpu);
  printf("    exchange interval: %d steps.\n", config_.exchange_interval);
  printf("    temperatures (K):");
  for (const double t : config_.temperatures)
    printf(" %g", t);
  printf("\n    pressures (GPa), one row per temperature:\n");
  for (int t = 0; t < num_t(); ++t) {
    printf("      %g K:", config_.temperatures[t]);
    for (int p = 0; p < num_p(); ++p)
      printf(" %g", config_.pressure_table[t * num_p() + p]);
    printf("\n");
  }
  printf("    thermostat: BDP, tau_T = %g steps.\n", barostat_input_.temperature_coupling);
  printf(
    "    barostat: stochastic cell rescaling, couple %s, bulk modulus %g GPa, tau_p = %g "
    "steps, update every %d steps.\n",
    barostat_input_.couple_name.c_str(),
    barostat_input_.bulk_modulus,
    barostat_input_.pressure_coupling,
    barostat_input_.interval);
  if (config_.equilibration_steps > 0)
    printf("    independent NPT equilibration: %d steps.\n", config_.equilibration_steps);
  if (config_.resume)
    printf("    resumed from %s at step %lld.\n", config_.resume_prefix.c_str(), completed_steps_);
  fflush(stdout);
}

void REMD_TP_Driver::build_phases()
{
  // Even T pairs, even P pairs, odd T pairs, odd P pairs. A dimension of size 1
  // has no pairs; a dimension of size 2 has only the (0, 1) pair.
  const int sizes[2] = {num_t(), num_p()};
  for (int parity = 0; parity < 2; ++parity) {
    for (int d = 0; d < 2; ++d) {
      if (sizes[d] < 2)
        continue;
      phases_.emplace_back(static_cast<Direction>(d), sizes[d] == 2 ? 0 : parity);
    }
  }
}

void REMD_TP_Driver::set_condition(Replica_Slot& slot, const int label)
{
  slot.temperature_label = label;
  slot.target_temperature = temperature_of(label);
  slot.thermostat->temperature = slot.target_temperature;
  slot.barostat->target_pressure = pressure_of(label) / PRESSURE_UNIT_CONVERSION;
}

void REMD_TP_Driver::initialize_slot(Replica_Slot& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  if (config_.resume)
    read_restart_xyz(slot);

  allocate_memory_gpu(slot.group, slot.atom, slot.thermo);
  const int label = replica_to_label_[slot.replica_id];
  if (!config_.resume) {
    generate_canonical_velocities(
      temperature_of(label),
      slot.atom.cpu_mass,
      slot.atom.cpu_velocity_per_atom,
      config_.internal_seed + 104729U * (slot.replica_id + 1U));
  }
  slot.atom.velocity_per_atom.copy_from_host_async(
    slot.atom.cpu_velocity_per_atom.data(), slot.stream.get());
  slot.stream.synchronize();

  slot.thermostat.reset(new Ensemble_BDP_Replica(
    temperature_of(label), barostat_input_.temperature_coupling));
  slot.thermostat->removed_degrees_of_freedom = 3;
  slot.barostat.reset(new SCR_Barostat(slot.device_id));
  SCR_Barostat& barostat = *slot.barostat;
  barostat.compressibility = PRESSURE_UNIT_CONVERSION / barostat_input_.bulk_modulus;
  barostat.tau_p = barostat_input_.pressure_coupling;
  barostat.interval = barostat_input_.interval;
  for (int d = 0; d < 3; ++d)
    barostat.couple[d] = barostat_input_.couple[d];
  set_condition(slot, label);

  if (config_.resume) {
    if (!slot.thermostat->import_rng_state(restart_bdp_states_[slot.replica_id]))
      PRINT_INPUT_ERROR("Invalid BDP RNG state in the REMD-TP restart.");
    if (!barostat.import_state(restart_scr_states_[slot.replica_id]))
      PRINT_INPUT_ERROR("Invalid SCR barostat state in the REMD-TP restart.");
    if (barostat.step_counter < completed_steps_ ||
        barostat.pending != (barostat.step_counter > 0 &&
                             barostat.step_counter % barostat.interval == 0))
      PRINT_INPUT_ERROR("Inconsistent SCR step counter or pending update in the REMD-TP restart.");
  } else {
    slot.thermostat->seed_rng(config_.internal_seed + 130363U * (slot.replica_id + 1U) + 17U);
    barostat.rng.seed(config_.internal_seed + 611953U * (slot.replica_id + 1U) + 29U);
  }
}

void REMD_TP_Driver::equilibrate()
{
  if (config_.equilibration_steps <= 0)
    return;
  printf("    Equilibrating all replicas independently for %d NPT steps.\n",
         config_.equilibration_steps);
  fflush(stdout);
  const int progress_interval = std::max(1, config_.equilibration_steps / 10);
  int completed = 0;
  while (completed < config_.equilibration_steps) {
    const int steps = std::min(progress_interval, config_.equilibration_steps - completed);
    runtime_.advance_all(steps);
    completed += steps;
    printf("    %d REMD-TP equilibration steps completed.\n", completed);
    fflush(stdout);
  }
}

static __global__ void gpu_scale_remd_tp_velocity(
  const int n, const double factor, double* vx, double* vy, double* vz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    vx[i] *= factor;
    vy[i] *= factor;
    vz[i] *= factor;
  }
}

void REMD_TP_Driver::scale_velocity(Replica_Slot& slot, const double factor)
{
  CHECK(gpuSetDevice(slot.device_id));
  const int n = slot.atom.number_of_atoms;
  double* v = slot.atom.velocity_per_atom.data();
  gpu_scale_remd_tp_velocity<<<(n - 1) / 128 + 1, 128, 0, slot.stream.get()>>>(
    n, factor, v, v + n, v + 2 * n);
  GPU_CHECK_KERNEL
}

void REMD_TP_Driver::attempt_pair(
  const long long step, const Direction direction, const int label_a, const int label_b, FILE* file)
{
  const int replica_i = label_to_replica_[label_a];
  const int replica_j = label_to_replica_[label_b];
  Replica_Slot& slot_i = *runtime_.slots()[replica_i];
  Replica_Slot& slot_j = *runtime_.slots()[replica_j];
  const double beta_a = beta_of(label_a);
  const double beta_b = beta_of(label_b);
  const double p_a = pressure_of(label_a) / PRESSURE_UNIT_CONVERSION;
  const double p_b = pressure_of(label_b) / PRESSURE_UNIT_CONVERSION;
  const double e_i = slot_i.potential_energy;
  const double e_j = slot_j.potential_energy;
  const double v_i = slot_i.box.get_volume();
  const double v_j = slot_j.box.get_volume();
  const double log_ratio = (beta_a - beta_b) * (e_i - e_j) + (beta_a * p_a - beta_b * p_b) * (v_i - v_j);
  const double probability = log_ratio >= 0.0 ? 1.0 : std::exp(log_ratio);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const bool accepted = log_ratio >= 0.0 || uniform(exchange_rng_) < probability;
  ++attempts_[direction][label_a];
  probability_sum_[direction][label_a] += probability;

  fprintf(
    file,
    "%lld %c %d %d %d %d %.10g %.10g %.10g %.10g %.16g %.16g %.16g %.16g %.10g %.10g %d\n",
    step,
    direction == along_temperature ? 'T' : 'P',
    replica_i,
    replica_j,
    label_a,
    label_b,
    temperature_of(label_a),
    temperature_of(label_b),
    pressure_of(label_a),
    pressure_of(label_b),
    e_i,
    e_j,
    v_i,
    v_j,
    log_ratio,
    probability,
    accepted ? 1 : 0);
  if (!accepted)
    return;

  ++accepts_[direction][label_a];
  if (direction == along_temperature) {
    scale_velocity(slot_i, std::sqrt(temperature_of(label_b) / temperature_of(label_a)));
    scale_velocity(slot_j, std::sqrt(temperature_of(label_a) / temperature_of(label_b)));
  }
  set_condition(slot_i, label_b);
  set_condition(slot_j, label_a);
  if (direction == along_temperature) {
    // Refresh the kinetic pressure after rescaling, without dropping an update
    // or consuming any SCR random numbers. The same stream orders both kernels.
    if (slot_i.barostat->pending)
      replica::refresh_barostat_pressure(slot_i);
    if (slot_j.barostat->pending)
      replica::refresh_barostat_pressure(slot_j);
  }
  replica_to_label_[replica_i] = label_b;
  replica_to_label_[replica_j] = label_a;
  label_to_replica_[label_a] = replica_j;
  label_to_replica_[label_b] = replica_i;
}

void REMD_TP_Driver::attempt_exchanges(const long long step, FILE* file)
{
  if (phases_.empty())
    return;
  const auto phase = phases_[exchange_count_ % phases_.size()];
  if (phase.first == along_temperature) {
    for (int p = 0; p < num_p(); ++p)
      for (int t = phase.second; t + 1 < num_t(); t += 2)
        attempt_pair(step, along_temperature, label_of(t, p), label_of(t + 1, p), file);
  } else {
    for (int t = 0; t < num_t(); ++t)
      for (int p = phase.second; p + 1 < num_p(); p += 2)
        attempt_pair(step, along_pressure, label_of(t, p), label_of(t, p + 1), file);
  }
  ++exchange_count_;
  fflush(file);
}

void REMD_TP_Driver::write_conditions() const
{
  FILE* file = my_fopen("remd_tp_conditions.out", "w");
  fprintf(file, "# REMD-TP conditions: label = p_index * num_temperatures + t_index\n");
  fprintf(file, "# num_temperatures %d num_pressures %d\n", num_t(), num_p());
  fprintf(file, "# label t_index p_index temperature_K pressure_GPa\n");
  for (int label = 0; label < config_.replicas; ++label)
    fprintf(file, "%d %d %d %.16g %.16g\n", label, label % num_t(), label / num_t(),
            temperature_of(label), pressure_of(label));
  fclose(file);
}

void REMD_TP_Driver::write_status(const long long step, FILE* file) const
{
  fprintf(file, "%lld", step);
  for (const int label : replica_to_label_)
    fprintf(file, " %d", label);
  fprintf(file, "\n");
  fflush(file);
}

void REMD_TP_Driver::write_thermo(const long long step, FILE* file) const
{
  fprintf(file, "%lld", step);
  for (const Replica_Slot* slot : runtime_.slots())
    fprintf(file, " %d %.16g %.16g", slot->temperature_label, slot->potential_energy,
            slot->box.get_volume());
  fprintf(file, "\n");
  fflush(file);
}

void REMD_TP_Driver::write_pair_report() const
{
  FILE* file = my_fopen("remd_tp_pairs.out", "w");
  fprintf(file, "# REMD-TP neighbour-pair exchange statistics\n");
  fprintf(file, "# direction label_a label_b T_a T_b P_a P_b attempts accepted "
                "acceptance_ratio average_probability\n");
  for (int d = 0; d < 2; ++d) {
    for (int label_a = 0; label_a < config_.replicas; ++label_a) {
      const int t = label_a % num_t();
      const int p = label_a / num_t();
      int label_b = -1;
      if (d == along_temperature && t + 1 < num_t())
        label_b = label_of(t + 1, p);
      if (d == along_pressure && p + 1 < num_p())
        label_b = label_of(t, p + 1);
      if (label_b < 0)
        continue;
      const long long n = attempts_[d][label_a];
      fprintf(file, "%c %d %d %.10g %.10g %.10g %.10g %lld %lld %.6f %.6f\n",
              d == along_temperature ? 'T' : 'P', label_a, label_b,
              temperature_of(label_a), temperature_of(label_b),
              pressure_of(label_a), pressure_of(label_b), n, accepts_[d][label_a],
              n > 0 ? static_cast<double>(accepts_[d][label_a]) / n : 0.0,
              n > 0 ? probability_sum_[d][label_a] / n : 0.0);
    }
  }
  fclose(file);
}

void REMD_TP_Driver::open_trajectory_files()
{
  if (!config_.dump_xyz)
    return;
  trajectory_files_.resize(config_.replicas, nullptr);
  for (int r = 0; r < config_.replicas; ++r)
    trajectory_files_[r] = my_fopen(
      trajectory_filename(config_.dump_prefix, r).c_str(), config_.resume ? "a" : "w");
}

void REMD_TP_Driver::close_trajectory_files()
{
  for (FILE* file : trajectory_files_) {
    if (file)
      fclose(file);
  }
  trajectory_files_.clear();
}

void REMD_TP_Driver::write_xyz_header(
  FILE* file, const Replica_Slot& slot, const long long step, const bool restart) const
{
  const double* h = slot.box.cpu_h;
  fprintf(file, "%d\n", slot.atom.number_of_atoms);
  fprintf(file,
          "Time=%.8f step=%lld replica=%d condition_label=%d target_temperature=%.17g "
          "target_pressure=%.17g potential_energy=%.17g ",
          step * time_step_ * TIME_UNIT_CONVERSION, step, slot.replica_id, slot.temperature_label,
          slot.target_temperature, pressure_of(slot.temperature_label), slot.potential_energy);
  fprintf(file, "pbc=\"%c %c %c\" ", slot.box.pbc_x ? 'T' : 'F', slot.box.pbc_y ? 'T' : 'F',
          slot.box.pbc_z ? 'T' : 'F');
  fprintf(file, "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" ", h[0], h[3],
          h[6], h[1], h[4], h[7], h[2], h[5], h[8]);
  if (restart)
    fprintf(file, "replica_mode=remd_tp Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n");
  else
    fprintf(file, "Properties=species:S:1:pos:R:3%s\n", config_.dump_velocity ? ":vel:R:3" : "");
}

void REMD_TP_Driver::write_trajectories(const long long step)
{
  if (!config_.dump_xyz)
    return;
  const int n = source_atom_.number_of_atoms;
  for (Replica_Slot* slot : runtime_.slots()) {
    CHECK(gpuSetDevice(slot->device_id));
    slot->stream.synchronize();
    slot->atom.position_per_atom.copy_to_host(slot->atom.cpu_position_per_atom.data());
    if (config_.dump_velocity)
      slot->atom.velocity_per_atom.copy_to_host(slot->atom.cpu_velocity_per_atom.data());
    FILE* file = trajectory_files_[slot->replica_id];
    write_xyz_header(file, *slot, step, false);
    for (int i = 0; i < n; ++i) {
      fprintf(file, "%s %.10g %.10g %.10g", source_atom_.cpu_atom_symbol[i].c_str(),
              slot->atom.cpu_position_per_atom[i], slot->atom.cpu_position_per_atom[i + n],
              slot->atom.cpu_position_per_atom[i + 2 * n]);
      if (config_.dump_velocity)
        fprintf(file, " %.10g %.10g %.10g",
                slot->atom.cpu_velocity_per_atom[i] / TIME_UNIT_CONVERSION,
                slot->atom.cpu_velocity_per_atom[i + n] / TIME_UNIT_CONVERSION,
                slot->atom.cpu_velocity_per_atom[i + 2 * n] / TIME_UNIT_CONVERSION);
      fprintf(file, "\n");
    }
    fflush(file);
  }
}

void REMD_TP_Driver::write_restart()
{
  const long long final_step = completed_steps_ + number_of_steps_;
  const int n = source_atom_.number_of_atoms;
  for (Replica_Slot* slot : runtime_.slots()) {
    CHECK(gpuSetDevice(slot->device_id));
    slot->stream.synchronize();
    slot->atom.position_per_atom.copy_to_host(slot->atom.cpu_position_per_atom.data());
    slot->atom.velocity_per_atom.copy_to_host(slot->atom.cpu_velocity_per_atom.data());
    const std::string filename =
      "remd_tp_replica_" + std::to_string(slot->replica_id) + "_restart.xyz";
    FILE* file = my_fopen(filename.c_str(), "w");
    write_xyz_header(file, *slot, final_step, true);
    for (int i = 0; i < n; ++i) {
      fprintf(file, "%s %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
              source_atom_.cpu_atom_symbol[i].c_str(), slot->atom.cpu_position_per_atom[i],
              slot->atom.cpu_position_per_atom[i + n], slot->atom.cpu_position_per_atom[i + 2 * n],
              source_atom_.cpu_mass[i],
              slot->atom.cpu_velocity_per_atom[i] / TIME_UNIT_CONVERSION,
              slot->atom.cpu_velocity_per_atom[i + n] / TIME_UNIT_CONVERSION,
              slot->atom.cpu_velocity_per_atom[i + 2 * n] / TIME_UNIT_CONVERSION);
    }
    fclose(file);
  }

  FILE* file = my_fopen("remd_tp_restart.meta", "w");
  fprintf(file, "# GPUMD REMD-TP restart metadata\n");
  fprintf(file, "format_version %d\n", restart_format_version);
  fprintf(file, "replicas %d\n", config_.replicas);
  fprintf(file, "completed_steps %lld\n", final_step);
  fprintf(file, "exchange_count %lld\n", exchange_count_);
  fprintf(file, "exchange_interval %d\n", config_.exchange_interval);
  fprintf(file, "time_step %.17g\n", time_step_);
  fprintf(file, "potential_hash %llu\n", potential_hash_);
  fprintf(file, "barostat %.17g %.17g %.17g %d %s\n", barostat_input_.temperature_coupling,
          barostat_input_.bulk_modulus, barostat_input_.pressure_coupling,
          barostat_input_.interval, barostat_input_.couple_name.c_str());
  fprintf(file, "temperatures");
  for (const double t : config_.temperatures)
    fprintf(file, " %.17g", t);
  fprintf(file, "\npressure_table");
  for (const double p : config_.pressure_table)
    fprintf(file, " %.17g", p);
  fprintf(file, "\nreplica_to_label");
  for (const int label : replica_to_label_)
    fprintf(file, " %d", label);
  for (int d = 0; d < 2; ++d) {
    fprintf(file, "\npair_statistics %d", d);
    for (int label = 0; label < config_.replicas; ++label)
      fprintf(file, " %lld %lld %.17g", attempts_[d][label], accepts_[d][label],
              probability_sum_[d][label]);
  }
  std::ostringstream exchange_state;
  exchange_state << exchange_rng_;
  fprintf(file, "\nexchange_rng %s\n", exchange_state.str().c_str());
  for (const Replica_Slot* slot : runtime_.slots()) {
    fprintf(file, "bdp_rng %d %s\n", slot->replica_id,
            slot->thermostat->export_rng_state().c_str());
    fprintf(file, "scr_state %d %s\n", slot->replica_id, slot->barostat->export_state().c_str());
  }
  fclose(file);
}

void REMD_TP_Driver::read_restart_metadata()
{
  const std::string filename = config_.resume_prefix + "_restart.meta";
  std::ifstream file_input(filename);
  if (!file_input.is_open())
    PRINT_INPUT_ERROR("Cannot open the REMD-TP restart metadata file.");
  const int n = config_.replicas;
  std::vector<bool> has_bdp(n, false), has_scr(n, false);
  bool has_statistics[2] = {false, false};
  std::set<std::string> seen_keys;
  std::string line;
  while (std::getline(file_input, line)) {
    std::istringstream input(line);
    std::string key;
    if (!(input >> key) || key[0] == '#')
      continue;
    if (key != "bdp_rng" && key != "scr_state" && key != "pair_statistics" &&
        !seen_keys.insert(key).second)
      PRINT_INPUT_ERROR("Duplicate key in the REMD-TP restart metadata.");
    if (key == "format_version") {
      int value = 0;
      if (!(input >> value) || value != restart_format_version)
        PRINT_INPUT_ERROR("Unsupported REMD-TP restart format_version.");
    } else if (key == "replicas") {
      int value = 0;
      if (!(input >> value) || value != n)
        PRINT_INPUT_ERROR("REMD-TP restart replicas do not match the input.");
    } else if (key == "completed_steps") {
      if (!(input >> completed_steps_) || completed_steps_ < 0)
        PRINT_INPUT_ERROR("REMD-TP restart completed_steps is invalid.");
    } else if (key == "exchange_count") {
      if (!(input >> exchange_count_) || exchange_count_ < 0)
        PRINT_INPUT_ERROR("REMD-TP restart exchange_count is invalid.");
    } else if (key == "exchange_interval") {
      int value = 0;
      if (!(input >> value) || value != config_.exchange_interval)
        PRINT_INPUT_ERROR("REMD-TP restart exchange_interval does not match the input.");
    } else if (key == "time_step") {
      double value = 0.0;
      if (!(input >> value) || !values_match(value, time_step_))
        PRINT_INPUT_ERROR("REMD-TP restart time_step does not match the input.");
    } else if (key == "potential_hash") {
      unsigned long long value = 0;
      if (!(input >> value) || value != potential_hash_)
        PRINT_INPUT_ERROR("REMD-TP restart potential files do not match the input.");
    } else if (key == "barostat") {
      double tau_t = 0.0, modulus = 0.0, tau_p = 0.0;
      int interval = 0;
      std::string couple;
      if (!(input >> tau_t >> modulus >> tau_p >> interval >> couple) ||
          !values_match(tau_t, barostat_input_.temperature_coupling) ||
          !values_match(modulus, barostat_input_.bulk_modulus) ||
          !values_match(tau_p, barostat_input_.pressure_coupling) ||
          interval != barostat_input_.interval || couple != barostat_input_.couple_name)
        PRINT_INPUT_ERROR("REMD-TP restart npt_scr parameters do not match the input.");
    } else if (key == "temperatures" || key == "pressure_table") {
      const std::vector<double>& expected =
        key == "temperatures" ? config_.temperatures : config_.pressure_table;
      for (const double e : expected) {
        double value = 0.0;
        if (!(input >> value) || !values_match(value, e))
          PRINT_INPUT_ERROR("REMD-TP restart conditions do not match the input.");
      }
    } else if (key == "replica_to_label") {
      std::vector<bool> seen(n, false);
      for (int r = 0; r < n; ++r) {
        int label = -1;
        if (!(input >> label) || label < 0 || label >= n || seen[label])
          PRINT_INPUT_ERROR("REMD-TP restart replica_to_label is not a permutation.");
        seen[label] = true;
        replica_to_label_[r] = label;
        label_to_replica_[label] = r;
      }
    } else if (key == "pair_statistics") {
      int d = -1;
      if (!(input >> d) || d < 0 || d > 1 || has_statistics[d])
        PRINT_INPUT_ERROR("REMD-TP restart pair_statistics is invalid.");
      has_statistics[d] = true;
      for (int label = 0; label < n; ++label) {
        if (!(input >> attempts_[d][label] >> accepts_[d][label] >> probability_sum_[d][label]))
          PRINT_INPUT_ERROR("REMD-TP restart pair_statistics is incomplete.");
        if (attempts_[d][label] < 0 || accepts_[d][label] < 0 ||
            accepts_[d][label] > attempts_[d][label] ||
            !std::isfinite(probability_sum_[d][label]) || probability_sum_[d][label] < 0.0 ||
            probability_sum_[d][label] > static_cast<double>(attempts_[d][label]))
          PRINT_INPUT_ERROR("REMD-TP restart pair_statistics is invalid.");
      }
    } else if (key == "exchange_rng") {
      std::string state;
      std::getline(input, state);
      std::istringstream state_input(trim_left(state));
      std::string trailing;
      if (!(state_input >> exchange_rng_) || (state_input >> trailing))
        PRINT_INPUT_ERROR("REMD-TP restart exchange_rng is invalid.");
    } else if (key == "bdp_rng" || key == "scr_state") {
      int r = -1;
      if (!(input >> r) || r < 0 || r >= n)
        PRINT_INPUT_ERROR("REMD-TP restart replica id is invalid.");
      std::string state;
      std::getline(input, state);
      if (key == "bdp_rng") {
        if (has_bdp[r] || trim_left(state).empty())
          PRINT_INPUT_ERROR("Duplicate or empty bdp_rng in the REMD-TP restart.");
        restart_bdp_states_[r] = trim_left(state);
        has_bdp[r] = true;
      } else {
        if (has_scr[r] || trim_left(state).empty())
          PRINT_INPUT_ERROR("Duplicate or empty scr_state in the REMD-TP restart.");
        restart_scr_states_[r] = trim_left(state);
        has_scr[r] = true;
      }
    } else {
      PRINT_INPUT_ERROR("Unknown key in the REMD-TP restart metadata.");
    }
    std::string trailing;
    if (input >> trailing)
      PRINT_INPUT_ERROR("Unexpected trailing data in the REMD-TP restart metadata.");
  }
  const char* required_keys[] = {
    "format_version", "replicas", "completed_steps", "exchange_count", "exchange_interval",
    "time_step", "potential_hash", "barostat", "temperatures", "pressure_table",
    "replica_to_label", "exchange_rng"};
  for (const char* key : required_keys) {
    if (seen_keys.count(key) == 0)
      PRINT_INPUT_ERROR("REMD-TP restart metadata is incomplete.");
  }
  if (!file_input.eof() || !has_statistics[0] || !has_statistics[1] ||
      std::find(has_bdp.begin(), has_bdp.end(), false) != has_bdp.end() ||
      std::find(has_scr.begin(), has_scr.end(), false) != has_scr.end())
    PRINT_INPUT_ERROR("REMD-TP restart metadata is incomplete.");
  if (exchange_count_ != completed_steps_ / config_.exchange_interval)
    PRINT_INPUT_ERROR("REMD-TP restart exchange_count does not match completed_steps.");
  for (int d = 0; d < 2; ++d)
    for (int label = 0; label < n; ++label)
      if (attempts_[d][label] > exchange_count_)
        PRINT_INPUT_ERROR("REMD-TP restart pair_statistics exceeds exchange_count.");
}

void REMD_TP_Driver::read_restart_xyz(Replica_Slot& slot)
{
  const std::string filename =
    config_.resume_prefix + "_replica_" + std::to_string(slot.replica_id) + "_restart.xyz";
  std::ifstream input(filename);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open an REMD-TP replica restart xyz file.");
  int number_of_atoms = 0;
  if (!(input >> number_of_atoms) || number_of_atoms != source_atom_.number_of_atoms)
    PRINT_INPUT_ERROR("REMD-TP restart atom count does not match model.xyz.");
  input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  std::string comment;
  if (!std::getline(input, comment))
    PRINT_INPUT_ERROR("Cannot read the REMD-TP restart xyz comment line.");
  replica::validate_xyz_pbc(comment, source_box_, "REMD-TP");

  std::string value;
  int file_replica = -1;
  int label = -1;
  long long step = -1;
  double target_temperature = 0.0, target_pressure = 0.0;
  if (!extract_token_value(comment, "replica_mode", value) || value != "remd_tp" ||
      !extract_token_value(comment, "Properties", value) ||
      value != "species:S:1:pos:R:3:mass:R:1:vel:R:3" ||
      !extract_token_value(comment, "replica", value) || !parse_scalar(value, file_replica) ||
      file_replica != slot.replica_id ||
      !extract_token_value(comment, "condition_label", value) || !parse_scalar(value, label) ||
      label != replica_to_label_[slot.replica_id] ||
      !extract_token_value(comment, "target_temperature", value) ||
      !parse_scalar(value, target_temperature) ||
      !values_match(target_temperature, temperature_of(label)) ||
      !extract_token_value(comment, "target_pressure", value) ||
      !parse_scalar(value, target_pressure) || !values_match(target_pressure, pressure_of(label)) ||
      !extract_token_value(comment, "step", value) || !parse_scalar(value, step) ||
      step != completed_steps_ || !extract_token_value(comment, "potential_energy", value) ||
      !parse_scalar(value, slot.potential_energy) || !std::isfinite(slot.potential_energy))
    PRINT_INPUT_ERROR("REMD-TP restart xyz metadata does not match the restart mapping.");
  if (!extract_quoted_value(comment, "Lattice", value))
    PRINT_INPUT_ERROR("REMD-TP restart xyz is missing Lattice.");
  std::istringstream lattice(value);
  const int transpose[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
  for (int i = 0; i < 9; ++i) {
    double h = 0.0;
    if (!(lattice >> h) || !std::isfinite(h))
      PRINT_INPUT_ERROR("REMD-TP restart xyz Lattice is invalid.");
    slot.box.cpu_h[transpose[i]] = h;
  }
  std::string trailing;
  if (lattice >> trailing)
    PRINT_INPUT_ERROR("REMD-TP restart xyz has too many Lattice values.");
  validate_barostat_box(slot.box);
  for (int d = 0; d < 3; ++d) {
    if (!barostat_input_.couple[d] &&
        !values_match(slot.box.cpu_h[d * 4], source_box_.cpu_h[d * 4]))
      PRINT_INPUT_ERROR("REMD-TP restart uncoupled box length does not match model.xyz.");
  }
  slot.box.get_inverse();
  for (int i = 9; i < 18; ++i) {
    if (!std::isfinite(slot.box.cpu_h[i]))
      PRINT_INPUT_ERROR("REMD-TP restart xyz inverse Lattice is invalid.");
  }

  const int n = number_of_atoms;
  for (int i = 0; i < n; ++i) {
    std::string symbol;
    double mass = 0.0;
    std::string line;
    if (!std::getline(input, line))
      PRINT_INPUT_ERROR("REMD-TP restart xyz atom data is incomplete.");
    std::istringstream row(line);
    if (!(row >> symbol >> slot.atom.cpu_position_per_atom[i] >>
          slot.atom.cpu_position_per_atom[i + n] >> slot.atom.cpu_position_per_atom[i + 2 * n] >>
          mass >> slot.atom.cpu_velocity_per_atom[i] >> slot.atom.cpu_velocity_per_atom[i + n] >>
          slot.atom.cpu_velocity_per_atom[i + 2 * n]) || (row >> trailing))
      PRINT_INPUT_ERROR("REMD-TP restart xyz atom data is incomplete.");
    if (symbol != source_atom_.cpu_atom_symbol[i] || !values_match(mass, source_atom_.cpu_mass[i]))
      PRINT_INPUT_ERROR("REMD-TP restart xyz species or mass does not match model.xyz.");
    for (int d = 0; d < 3; ++d) {
      const int index = i + d * n;
      slot.atom.cpu_velocity_per_atom[index] *= TIME_UNIT_CONVERSION;
      if (!std::isfinite(slot.atom.cpu_position_per_atom[index]) ||
          !std::isfinite(slot.atom.cpu_velocity_per_atom[index]))
        PRINT_INPUT_ERROR("REMD-TP restart xyz contains a non-finite atom value.");
    }
  }
  if (input >> trailing)
    PRINT_INPUT_ERROR("REMD-TP restart xyz contains extra atom data or frames.");
}

void REMD_TP_Driver::run()
{
  validate_input();
  const auto initialization_begin = std::chrono::high_resolution_clock::now();
  runtime_.initialize(
    config_.replicas, config_.replicas_per_gpu, [this](Replica_Slot& slot) { initialize_slot(slot); });
  runtime_.start_workers();
  if (!config_.resume)
    equilibrate();
  const auto initialization_end = std::chrono::high_resolution_clock::now();

  write_conditions();
  const bool append_exchange = config_.resume && file_has_content("remd_tp_exchange.out");
  const bool append_status = config_.resume && file_has_content("remd_tp_status.out");
  const bool append_thermo = config_.resume && file_has_content("remd_tp_thermo.out");
  const char* mode = config_.resume ? "a" : "w";
  FILE* exchange_file = my_fopen("remd_tp_exchange.out", mode);
  FILE* status_file = my_fopen("remd_tp_status.out", mode);
  FILE* thermo_file = my_fopen("remd_tp_thermo.out", mode);
  if (!append_exchange) {
    fprintf(exchange_file, "%s\n", exchange_output_marker);
    fprintf(exchange_file, "# States are recorded before swapping. Units: K, GPa, eV, A^3.\n");
    fprintf(exchange_file,
            "# step direction replica_i replica_j label_a label_b T_a T_b P_a P_b E_i E_j V_i "
            "V_j log_acceptance_ratio acceptance_probability accepted\n");
  }
  if (!append_status) {
    fprintf(status_file, "%s\n", status_output_marker);
    fprintf(status_file, "# step condition_label_of_replica_0 ... (see remd_tp_conditions.out)\n");
  }
  if (!append_thermo) {
    fprintf(thermo_file, "%s\n", thermo_output_marker);
    fprintf(thermo_file, "# Replica states after any exchange at the reported step.\n");
    fprintf(thermo_file,
            "# step, then per replica: condition_label potential_energy(eV) volume(A^3)\n");
  }
  open_trajectory_files();
  if (!append_status)
    write_status(completed_steps_, status_file);
  if (!append_thermo)
    write_thermo(completed_steps_, thermo_file);

  const auto production_begin = std::chrono::high_resolution_clock::now();
  const int progress_interval = std::max(1, number_of_steps_ / 10);
  int local_step = 0;
  while (local_step < number_of_steps_) {
    const long long global_step = completed_steps_ + local_step;
    int next_event =
      next_periodic_event(local_step, number_of_steps_, global_step, config_.exchange_interval);
    if (config_.dump_xyz)
      next_event = std::min(
        next_event,
        next_periodic_event(local_step, number_of_steps_, global_step, config_.dump_interval));
    next_event = std::min(
      next_event,
      std::min(number_of_steps_, ((local_step / progress_interval) + 1) * progress_interval));
    runtime_.advance_all(next_event - local_step);
    local_step = next_event;
    const long long step = completed_steps_ + local_step;

    const bool exchange_due = step % config_.exchange_interval == 0;
    const bool dump_due = config_.dump_xyz && step % config_.dump_interval == 0;
    if (exchange_due)
      attempt_exchanges(step, exchange_file);
    write_thermo(step, thermo_file);
    if (exchange_due || dump_due || local_step == number_of_steps_)
      write_status(step, status_file);
    if (dump_due)
      write_trajectories(step);
    if (local_step % progress_interval == 0 || local_step == number_of_steps_) {
      printf("    %d REMD-TP production steps completed.\n", local_step);
      fflush(stdout);
    }
  }
  const auto production_end = std::chrono::high_resolution_clock::now();

  runtime_.stop_workers();
  write_restart();
  write_pair_report();
  close_trajectory_files();
  fclose(exchange_file);
  fclose(status_file);
  fclose(thermo_file);

  const std::chrono::duration<double> init_time = initialization_end - initialization_begin;
  const std::chrono::duration<double> production_time = production_end - production_begin;
  print_line_1();
  printf("REMD-TP initialization/equilibration time = %g second.\n", init_time.count());
  printf("Time used for this REMD-TP production run = %g second.\n", production_time.count());
  printf("Speed of this REMD-TP run = %g atom*step/second.\n",
         source_atom_.number_of_atoms *
           (static_cast<double>(number_of_steps_) * config_.replicas / production_time.count()));
  print_line_2();
}

} // namespace

void replica::run_remd_tp(
  const Replica_Input& input, const RunInput& run_input,
  Atom& source_atom, Box& source_box, std::vector<Group>& source_group)
{
  REMD_TP_Driver driver(input, run_input, source_atom, source_box, source_group);
  driver.run();
}
