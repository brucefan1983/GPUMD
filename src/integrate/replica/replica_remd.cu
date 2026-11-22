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
Replica-exchange molecular dynamics. Each visible GPU owns one potential model.
Independent replica states on that GPU use separate streams. NVT exchanges
temperatures with the BDP thermostat; NPT exchanges temperature-pressure
conditions with the BDP thermostat and the SCR barostat.

Reference for temperature-pressure exchange:
T. Okabe, M. Kawata, Y. Okamoto, and M. Mikami, Chem. Phys. Lett. 335, 435 (2001).

Major contributors:
  Wenhao Luo: Initial implementation.
  Jiuyang Shi: NPT temperature-pressure exchange, proposed and first implemented
    as a standalone replica program.
------------------------------------------------------------------------------*/

#include "main_gpumd/velocity.cuh"
#include "model/read_xyz.cuh"
#include "replica.cuh"
#include "replica_common.cuh"
#include "replica_scr.cuh"
#include "replica_velocity.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
using namespace replica;

constexpr double target_acceptance = 0.2;
constexpr int restart_format_version = 5;     // existing NVT checkpoints
constexpr int npt_restart_format_version = 6; // ensemble and independent SCR states
constexpr const char* npt_exchange_marker = "# GPUMD REMD NPT exchange output format 1";
constexpr const char* npt_status_marker = "# GPUMD REMD NPT condition status output format 1";
constexpr const char* npt_thermo_marker = "# GPUMD REMD NPT thermodynamic output format 1";
constexpr const char* exchange_output_marker = "# GPUMD REMD exchange output format 1";
constexpr const char* status_output_marker = "# GPUMD REMD temperature status output format 1";
constexpr const char* thermo_output_marker = "# GPUMD REMD thermodynamic output format 1";

int next_periodic_event(
  const int local_step, const int final_local_step, const long long global_step, const int interval)
{
  const long long remainder = global_step % interval;
  const long long steps = remainder == 0 ? interval : interval - remainder;
  return static_cast<int>(
    std::min<long long>(final_local_step, static_cast<long long>(local_step) + steps));
}

struct REMD_Pair {
  int direction; // 0: temperature, 1: pressure
  int first;
  int second;
};

struct REMD_Config {
  enum Temperature_Style { geometric, linear };

  int replicas = 0;
  int replicas_per_gpu = 0;
  int exchange_interval = 0;
  int equilibration_steps = 0;
  int dump_interval = 0;
  int pcouple_interval = 10; // NPT: MD steps between barostat updates
  bool is_npt = false;
  bool resume = false;
  bool dump_xyz = false;
  bool dump_velocity = false;
  Replica_XYZ_Fields dump_fields;
  Temperature_Style temperature_style = geometric;
  std::string resume_prefix;
  std::string dump_prefix;
  std::vector<double> temperatures;
  int num_pressures = 1;
  std::vector<double> pressure_table; // GPa; row-major [temperature][pressure]
  std::vector<double> betas;
  unsigned int internal_seed = 1;

  void parse(
    const std::vector<std::string>& command,
    const std::vector<std::string>& dump_command,
    bool npt,
    double ensemble_pressure);

  int num_temperatures() const { return static_cast<int>(temperatures.size()); }
  int label(int t, int p) const { return p * num_temperatures() + t; }
  double temperature(int label) const { return temperatures[label % num_temperatures()]; }
  double pressure(int label) const
  {
    return is_npt ? pressure_table
                      [(label % num_temperatures()) * num_pressures + label / num_temperatures()]
                  : 0.0;
  }

  std::string validate() const
  {
    if (
      temperatures.empty() || num_pressures <= 0 ||
      static_cast<long long>(num_temperatures()) * num_pressures != replicas)
      return "REMD grid dimensions must multiply to the replica count.";
    for (size_t i = 0; i < temperatures.size(); ++i)
      if (
        !std::isfinite(temperatures[i]) || temperatures[i] <= 0 ||
        (i && temperatures[i] <= temperatures[i - 1]))
        return "REMD temperatures must be positive, finite and strictly increasing.";
    if (!is_npt && (num_pressures != 1 || !pressure_table.empty()))
      return "NVT REMD does not accept pressure conditions.";
    if (is_npt) {
      if (pressure_table.size() != static_cast<size_t>(replicas))
        return "NPT REMD needs one pressure for each condition.";
      for (int t = 0; t < num_temperatures(); ++t)
        for (int p = 0; p < num_pressures; ++p) {
          const double value = pressure_table[t * num_pressures + p];
          if (!std::isfinite(value) || (p && value <= pressure_table[t * num_pressures + p - 1]))
            return "REMD pressures must be finite and increasing within each temperature row.";
        }
    }
    return {};
  }

  // Exchange pairs of one round. Temperature and pressure edges alternate, each with
  // even and odd phases; an axis with two conditions has only the even phase.
  std::vector<REMD_Pair> pairs(long long round) const
  {
    std::vector<std::pair<int, int>> phases;
    const int sizes[2] = {num_temperatures(), num_pressures};
    for (int parity = 0; parity < 2; ++parity)
      for (int direction = 0; direction < 2; ++direction)
        if (sizes[direction] > 1)
          phases.emplace_back(direction, sizes[direction] == 2 ? 0 : parity);
    std::vector<REMD_Pair> result;
    if (phases.empty())
      return result;
    const auto phase = phases[round % phases.size()];
    if (phase.first == 0) {
      for (int p = 0; p < num_pressures; ++p)
        for (int t = phase.second; t + 1 < num_temperatures(); t += 2)
          result.push_back({0, label(t, p), label(t + 1, p)});
    } else {
      for (int t = 0; t < num_temperatures(); ++t)
        for (int p = phase.second; p + 1 < num_pressures; p += 2)
          result.push_back({1, label(t, p), label(t, p + 1)});
    }
    return result;
  }

  size_t statistic_index(const REMD_Pair& pair) const
  {
    return is_npt
             ? static_cast<size_t>(pair.direction) * num_temperatures() * num_pressures + pair.first
             : static_cast<size_t>(pair.first);
  }

  std::vector<long long> attempt_counts(long long rounds) const
  {
    const size_t count = static_cast<size_t>(num_temperatures()) * num_pressures;
    std::vector<long long> result(is_npt ? 2 * count : count - 1, 0);
    const int period = 2 * ((num_temperatures() > 1) + (num_pressures > 1));
    for (int phase = 0; phase < period; ++phase)
      for (const auto& pair : pairs(phase))
        result[statistic_index(pair)] += rounds / period + (phase < rounds % period);
    return result;
  }
};

// Logarithm of the acceptance ratio for exchanging the states i and j between the
// conditions a and b (pressures in eV/A^3).
double remd_log_ratio(
  const double temperature_a,
  const double temperature_b,
  const double pressure_a,
  const double pressure_b,
  const double energy_i,
  const double energy_j,
  const double volume_i,
  const double volume_j)
{
  const double beta_a = 1.0 / (K_B * temperature_a);
  const double beta_b = 1.0 / (K_B * temperature_b);
  return (beta_a - beta_b) * (energy_i - energy_j) +
         (beta_a * pressure_a - beta_b * pressure_b) * (volume_i - volume_j);
}

bool is_finite_real(const char* text, double* value)
{
  return is_valid_real(text, value) && std::isfinite(*value);
}

void read_ladder_file(const std::string& name, std::vector<double>& values)
{
  std::ifstream input(name);
  if (!input.is_open())
    PRINT_INPUT_ERROR("Cannot open the REMD condition file.");
  while (input.peek() != EOF) {
    const auto tokens = get_tokens(input);
    for (const auto& token : tokens) {
      double value = 0;
      if (!is_finite_real(token.c_str(), &value))
        PRINT_INPUT_ERROR("REMD conditions must be finite numbers.");
      values.push_back(value);
    }
  }
}

void REMD_Config::parse(
  const std::vector<std::string>& command,
  const std::vector<std::string>& dump_command,
  const bool npt,
  const double ensemble_pressure)
{
  is_npt = npt;
  if (command.size() < 3 || command[0] != "multi_replica" || command[1] != "remd")
    PRINT_INPUT_ERROR("Expected multi_replica remd followed by its options.");
  bool has_temperature = false, has_pressure = false;
  bool temperature_range = false, pressure_range = false;
  double t_min = 0, t_max = 0, p_min = 0, p_max = 0;
  int nt = 0;
  std::vector<double> pressures;
  std::string pressure_grid;
  for (size_t i = 2; i < command.size();) {
    const std::string option = command[i++];
    if (option == "replicas") {
      --i;
      parse_replicas_option(command, i, 2, "REMD", replicas, replicas_per_gpu);
    } else if (
      option == "exchange" || option == "equilibrate" || option == "equilibration" ||
      option == "equilibration_steps") {
      int& value = option == "exchange" ? exchange_interval : equilibration_steps;
      if (
        i >= command.size() || !is_valid_int(command[i++].c_str(), &value) ||
        (option == "exchange" ? value <= 0 : value < 0))
        PRINT_INPUT_ERROR("Invalid REMD exchange or equilibration interval.");
    } else if (option == "temp" || option == "pressure") {
      const bool temperature = option == "temp";
      if (!temperature && !is_npt)
        PRINT_INPUT_ERROR("Pressure conditions require ensemble npt_scr.");
      bool& seen = temperature ? has_temperature : has_pressure;
      if (seen || i >= command.size())
        PRINT_INPUT_ERROR("Specify each REMD condition source exactly once.");
      seen = true;
      bool& range = temperature ? temperature_range : pressure_range;
      double& minimum = temperature ? t_min : p_min;
      double& maximum = temperature ? t_max : p_max;
      int& count = temperature ? nt : num_pressures;
      auto& values = temperature ? temperatures : pressures;
      if (is_finite_real(command[i].c_str(), &minimum)) {
        range = true;
        if (++i >= command.size() || !is_finite_real(command[i++].c_str(), &maximum))
          PRINT_INPUT_ERROR("REMD range requires finite minimum and maximum values.");
        if (
          is_npt &&
          (i >= command.size() || !is_valid_int(command[i++].c_str(), &count) || count < 1))
          PRINT_INPUT_ERROR("NPT REMD ranges require a positive condition count.");
        if (
          is_npt &&
          ((count == 1 && !values_match(minimum, maximum)) || (count > 1 && maximum <= minimum)))
          PRINT_INPUT_ERROR("REMD range needs min < max, or min = max for one condition.");
        if (!is_npt && (minimum <= 0 || maximum <= minimum))
          PRINT_INPUT_ERROR("temp should satisfy 0 < Tmin < Tmax.");
      } else {
        read_ladder_file(command[i++], values);
        count = static_cast<int>(values.size());
      }
    } else if (option == "pressure_grid") {
      if (!is_npt || has_pressure || i >= command.size())
        PRINT_INPUT_ERROR("pressure_grid requires NPT REMD and one pressure source.");
      has_pressure = true;
      pressure_grid = command[i++];
    } else if (option == "pcouple_interval") {
      if (
        !is_npt || i >= command.size() || !is_valid_int(command[i++].c_str(), &pcouple_interval) ||
        pcouple_interval <= 0)
        PRINT_INPUT_ERROR("pcouple_interval requires ensemble npt_scr and a positive integer.");
    } else if (option == "spacing") {
      if (i >= command.size() || (command[i] != "geometric" && command[i] != "linear"))
        PRINT_INPUT_ERROR("spacing requires geometric or linear.");
      temperature_style = command[i++] == "linear" ? linear : geometric;
    } else if (option == "resume") {
      if (resume || i >= command.size())
        PRINT_INPUT_ERROR("resume requires one restart prefix.");
      resume = true;
      resume_prefix = command[i++];
    } else {
      PRINT_INPUT_ERROR("Unknown multi_replica option.");
    }
  }
  if (replicas < 2 || exchange_interval <= 0 || !has_temperature)
    PRINT_INPUT_ERROR("REMD requires replicas, exchange, and temp options.");
  if (is_npt && !has_pressure) {
    // Without a pressure ladder, every replica uses the npt_scr pressure.
    pressures.assign(1, ensemble_pressure);
    num_pressures = 1;
  }
  if (resume && equilibration_steps > 0)
    PRINT_INPUT_ERROR("REMD resume cannot be combined with equilibration.");
  if (!is_npt)
    nt = replicas;
  if (nt <= 0 || nt > replicas || num_pressures <= 0 || num_pressures > replicas)
    PRINT_INPUT_ERROR("Invalid REMD grid dimensions.");
  if (temperature_range) {
    temperatures.resize(nt);
    for (int t = 0; t < nt; ++t) {
      const double fraction = nt == 1 ? 0 : static_cast<double>(t) / (nt - 1);
      temperatures[t] = temperature_style == linear ? t_min + fraction * (t_max - t_min)
                                                    : t_min * std::pow(t_max / t_min, fraction);
    }
  }
  if (is_npt) {
    if (!pressure_grid.empty()) {
      std::ifstream input(pressure_grid);
      if (!input.is_open())
        PRINT_INPUT_ERROR("Cannot open the REMD pressure_grid file.");
      int rows = 0;
      while (input.peek() != EOF) {
        const auto tokens = get_tokens(input);
        if (tokens.empty())
          continue;
        if (rows >= nt || tokens.size() > static_cast<size_t>(replicas))
          PRINT_INPUT_ERROR("REMD pressure_grid dimensions exceed the replica count.");
        if (rows == 0)
          num_pressures = static_cast<int>(tokens.size());
        if (static_cast<int>(tokens.size()) != num_pressures)
          PRINT_INPUT_ERROR("REMD pressure_grid rows must have the same length.");
        for (const auto& token : tokens) {
          double value = 0;
          if (!is_finite_real(token.c_str(), &value))
            PRINT_INPUT_ERROR("REMD pressure_grid must contain finite pressures.");
          pressure_table.push_back(value);
        }
        ++rows;
      }
      if (rows != nt)
        PRINT_INPUT_ERROR("REMD pressure_grid requires one row per temperature.");
    } else {
      if (pressure_range) {
        pressures.resize(num_pressures);
        for (int p = 0; p < num_pressures; ++p)
          pressures[p] =
            p_min + (num_pressures == 1 ? 0 : static_cast<double>(p) / (num_pressures - 1)) *
                      (p_max - p_min);
      }
      for (int t = 0; t < nt; ++t)
        pressure_table.insert(pressure_table.end(), pressures.begin(), pressures.end());
    }
  }
  const std::string error = validate();
  if (!error.empty())
    PRINT_INPUT_ERROR(error.c_str());
  betas.resize(nt);
  for (int t = 0; t < nt; ++t)
    betas[t] = 1.0 / (K_B * temperatures[t]);
  if (!dump_command.empty()) {
    parse_dump_xyz(
      dump_command, "REMD", true, dump_interval, dump_prefix, dump_velocity, &dump_fields);
    dump_xyz = true;
  }
  internal_seed = draw_internal_seed();
}

class REMD_Driver
{
public:
  REMD_Driver(
    const replica::Replica_Input& input,
    const RunInput& run_input,
    Atom& source_atom,
    Box& source_box,
    std::vector<Group>& source_group);

  ~REMD_Driver();
  void run();

private:
  // ensemble npt_scr <T_1> <T_2> <T_coup> <p_hydro> <C_hydro> <p_coup>
  struct Barostat_Input {
    double temperature_1 = 0.0;        // K
    double temperature_2 = 0.0;        // K
    double temperature_coupling = 0.0; // MD steps
    double pressure = 0.0;             // GPa
    double bulk_modulus = 0.0;         // GPa
    double pressure_coupling = 0.0;    // MD steps
  };

  REMD_Config config_;
  Barostat_Input barostat_input_;
  std::vector<std::string> restart_scr_states_;
  long long restart_scr_counter_ = -1;
  const std::vector<std::vector<std::string>>& potential_commands_;
  const std::vector<std::string>& ensemble_command_;
  std::string kspace_method_;
  Atom& source_atom_;
  Box& source_box_;
  double time_step_;
  int number_of_steps_;
  double temperature_coupling_ = 0.0;
  unsigned long long potential_hash_ = 0;
  long long completed_steps_ = 0;
  long long exchange_count_ = 0;
  std::mt19937 exchange_rng_;
  std::vector<std::string> restart_bdp_states_;
  std::vector<int> replica_to_temperature_;
  std::vector<int> temperature_to_replica_;
  std::vector<long long> exchange_attempts_;
  std::vector<long long> exchange_accepts_;
  std::vector<double> exchange_probability_sum_;
  Replica_Runtime runtime_;
  std::vector<FILE*> trajectory_files_;

  void validate_input();
  void parse_npt_ensemble();
  double temperature_of(int label) const { return config_.temperature(label); }
  double pressure_of(int label) const { return config_.pressure(label); }
  std::string output_file(const char* kind) const
  {
    if (config_.is_npt)
      return std::string("remd_npt_") + kind + ".out";
    if (std::string(kind) == "status")
      return "remd_temperature_status.out";
    return std::string("remd_") + kind + ".out";
  }
  const char* exchange_marker() const
  {
    return config_.is_npt ? npt_exchange_marker : exchange_output_marker;
  }
  const char* status_marker() const
  {
    return config_.is_npt ? npt_status_marker : status_output_marker;
  }
  const char* thermo_marker() const
  {
    return config_.is_npt ? npt_thermo_marker : thermo_output_marker;
  }
  int stats_index(const REMD_Pair& pair) const
  {
    return static_cast<int>(config_.statistic_index(pair));
  }
  void write_npt_checkpoint(FILE* file) const;
  void read_npt_checkpoint(std::istream& input);
  void initialize_slot(Replica_Slot& slot);
  void set_temperature(Replica_Slot& slot, int temperature_label);
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
  const replica::Replica_Input& input,
  const RunInput& run_input,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group)
  : potential_commands_(input.potential_commands),
    ensemble_command_(input.ensemble_command),
    kspace_method_(input.kspace_method),
    source_atom_(source_atom),
    source_box_(source_box),
    time_step_(input.time_step),
    number_of_steps_(input.number_of_steps),
    runtime_(
      input.model_commands, run_input, source_atom, source_box, source_group, input.time_step)
{
  if (
    ensemble_command_.size() < 2 ||
    (ensemble_command_[1] != "nvt_bdp" && ensemble_command_[1] != "npt_scr"))
    PRINT_INPUT_ERROR("REMD requires ensemble nvt_bdp or npt_scr.");
  const bool npt = ensemble_command_[1] == "npt_scr";
  if (npt)
    parse_npt_ensemble();
  config_.parse(input.multi_replica_command, input.dump_xyz_command, npt, barostat_input_.pressure);
}

REMD_Driver::~REMD_Driver()
{
  runtime_.stop_workers();
  close_trajectory_files();
}

void REMD_Driver::parse_npt_ensemble()
{
  // Same parameters as the isotropic form of npt_scr in ordinary MD. The pressure
  // ladder, if any, is given by multi_replica remd.
  const auto& e = ensemble_command_;
  if (e.size() != 8)
    PRINT_INPUT_ERROR("NPT REMD supports only the isotropic npt_scr form: "
                      "ensemble npt_scr <T_1> <T_2> <T_coup> <p_hydro> <C_hydro> <p_coup>.");
  auto& b = barostat_input_;
  if (
    !is_finite_real(e[2].c_str(), &b.temperature_1) ||
    !is_finite_real(e[3].c_str(), &b.temperature_2) ||
    !is_finite_real(e[4].c_str(), &b.temperature_coupling) || b.temperature_coupling < 1.0)
    PRINT_INPUT_ERROR("Invalid npt_scr temperatures or T_coup (at least 1) for NPT REMD.");
  if (!is_finite_real(e[5].c_str(), &b.pressure))
    PRINT_INPUT_ERROR("npt_scr p_hydro should be a finite number (GPa).");
  if (!is_finite_real(e[6].c_str(), &b.bulk_modulus) || b.bulk_modulus <= 0.0)
    PRINT_INPUT_ERROR("npt_scr C_hydro should be positive (GPa).");
  // npt_scr does not change the box for an elastic modulus above 2000 GPa.
  if (b.bulk_modulus > 2.0e3)
    PRINT_INPUT_ERROR("NPT REMD requires an active barostat: C_hydro should not exceed 2000 GPa.");
  if (!is_finite_real(e[7].c_str(), &b.pressure_coupling) || b.pressure_coupling < 1.0)
    PRINT_INPUT_ERROR("npt_scr p_coup should be at least 1 step.");
}

void REMD_Driver::validate_input()
{
  if (number_of_steps_ <= 0)
    PRINT_INPUT_ERROR("REMD run steps should be positive.");
  if (source_atom_.number_of_atoms <= 0)
    PRINT_INPUT_ERROR("REMD requires a non-empty model.");
  potential_hash_ = validate_potential(potential_commands_, source_box_, "REMD", kspace_method_);
  double input_temperature_1 = 0.0;
  double input_temperature_2 = 0.0;
  if (config_.is_npt) {
    if (source_atom_.number_of_atoms < 2 || kspace_method_ != "none")
      PRINT_INPUT_ERROR("NPT REMD requires at least two atoms and an energy NEP model (no qNEP).");
    const auto& b = barostat_input_;
    temperature_coupling_ = b.temperature_coupling;
    if (
      !values_match(b.temperature_1, config_.temperatures.front()) ||
      !values_match(b.temperature_2, config_.temperatures.back()))
      PRINT_INPUT_ERROR(
        "npt_scr temperatures should match the first and last NPT REMD temperatures.");
    if (!values_match(
          b.pressure,
          *std::min_element(config_.pressure_table.begin(), config_.pressure_table.end())))
      PRINT_INPUT_ERROR("npt_scr p_hydro should match the lowest NPT REMD pressure.");
    if (b.pressure_coupling < 2.0 * config_.pcouple_interval)
      PRINT_INPUT_ERROR("npt_scr p_coup should be at least twice pcouple_interval.");
    if (!source_box_.pbc_x || !source_box_.pbc_y || !source_box_.pbc_z)
      PRINT_INPUT_ERROR("NPT REMD requires periodic boundaries in all directions.");
    for (int i = 0; i < 9; ++i)
      if (
        !std::isfinite(source_box_.cpu_h[i]) ||
        (i % 4 == 0 ? source_box_.cpu_h[i] <= 0 : source_box_.cpu_h[i] != 0))
        PRINT_INPUT_ERROR("NPT REMD requires a positive finite orthogonal box.");
    if (!std::isfinite(source_box_.get_volume()) || source_box_.get_volume() <= 0)
      PRINT_INPUT_ERROR("NPT REMD requires a positive finite volume.");
  } else {
    parse_nvt_bdp(
      ensemble_command_, "REMD", input_temperature_1, input_temperature_2, temperature_coupling_);
    if (
      !values_match(input_temperature_1, config_.temperatures.front()) ||
      !values_match(input_temperature_2, config_.temperatures.back()))
      PRINT_INPUT_ERROR(
        "REMD ensemble temperatures should match the first and last ladder temperatures.");
  }

  config_.replicas_per_gpu =
    resolve_replicas_per_gpu(config_.replicas, config_.replicas_per_gpu, "REMD");

  replica_to_temperature_.resize(config_.replicas);
  temperature_to_replica_.resize(config_.replicas);
  const int statistics = config_.is_npt ? 2 * config_.replicas : config_.replicas - 1;
  exchange_attempts_.assign(statistics, 0);
  exchange_accepts_.assign(statistics, 0);
  exchange_probability_sum_.assign(statistics, 0.0);
  if (config_.is_npt)
    restart_scr_states_.resize(config_.replicas);
  for (int r = 0; r < config_.replicas; ++r) {
    replica_to_temperature_[r] = r;
    temperature_to_replica_[r] = r;
  }

  if (config_.resume) {
    read_restart_metadata();
    validate_append_marker(output_file("exchange"), exchange_marker(), "REMD");
    validate_append_marker(output_file("status"), status_marker(), "REMD");
    validate_append_marker(output_file("thermo"), thermo_marker(), "REMD");
  } else {
    exchange_rng_.seed(config_.internal_seed);
  }

  printf(
    "%s replica-exchange molecular dynamics.\n",
    config_.is_npt ? "Temperature-pressure" : "Temperature");
  printf("    replicas: %d.\n", config_.replicas);
  printf("    replicas per GPU: %d.\n", config_.replicas_per_gpu);
  printf("    exchange interval: %d steps.\n", config_.exchange_interval);
  printf("    thermostat: BDP with tau_T = %g steps.\n", temperature_coupling_);
  printf(
    "    temperature range: %g K to %g K.\n",
    config_.temperatures.front(),
    config_.temperatures.back());
  if (config_.is_npt) {
    printf(
      "    barostat: isotropic SCR, bulk modulus %g GPa, tau_p = %g steps, update every %d "
      "steps.\n",
      barostat_input_.bulk_modulus,
      barostat_input_.pressure_coupling,
      config_.pcouple_interval);
    printf(
      "    grid: %d temperatures x %d pressures, pressures %g GPa to %g GPa.\n",
      config_.num_temperatures(),
      config_.num_pressures,
      *std::min_element(config_.pressure_table.begin(), config_.pressure_table.end()),
      *std::max_element(config_.pressure_table.begin(), config_.pressure_table.end()));
  }
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
  slot.target_temperature = temperature_of(temperature_label);
  if (slot.barostat)
    slot.barostat->set_target_pressure(pressure_of(temperature_label) / PRESSURE_UNIT_CONVERSION);
  slot.thermostat->temperature = slot.target_temperature;
}

void REMD_Driver::initialize_slot(Replica_Slot& slot)
{
  CHECK(gpuSetDevice(slot.device_id));
  if (config_.resume)
    read_restart_xyz(slot);

  allocate_memory_gpu(slot.group, slot.atom, slot.thermo);
  if (!config_.resume) {
    if (config_.is_npt) {
      Replica_Velocity::generate_canonical(
        temperature_of(slot.replica_id),
        slot.atom.cpu_mass,
        slot.atom.cpu_velocity_per_atom,
        config_.internal_seed + 104729U * (slot.replica_id + 1U));
    } else {
      Velocity velocity;
      velocity.initialize_cpu(
        false,
        config_.temperatures[slot.replica_id],
        slot.atom,
        true,
        static_cast<int>(config_.internal_seed + 104729U * (slot.replica_id + 1U)));
    }
  }
  slot.atom.velocity_per_atom.copy_from_host(slot.atom.cpu_velocity_per_atom.data());

  slot.thermostat.reset(new Ensemble_BDP_Replica(0.0, temperature_coupling_));
  if (config_.is_npt) {
    slot.thermostat->removed_degrees_of_freedom = 3;
    slot.barostat.reset(new SCR_Barostat(
      slot.device_id,
      slot.replica_id,
      barostat_input_.bulk_modulus,
      barostat_input_.pressure_coupling,
      config_.pcouple_interval,
      config_.internal_seed + 611953U * (slot.replica_id + 1U) + 29U));
    const SCR_Barostat& barostat = *slot.barostat;
    if (config_.resume) {
      if (
        !slot.barostat->import_state(restart_scr_states_[slot.replica_id]) ||
        barostat.step_counter() < completed_steps_)
        PRINT_INPUT_ERROR("Invalid SCR state in the REMD restart.");
      if (restart_scr_counter_ < 0)
        restart_scr_counter_ = barostat.step_counter();
      if (restart_scr_counter_ != barostat.step_counter())
        PRINT_INPUT_ERROR("REMD restart SCR counters differ between replicas.");
    }
  }
  set_temperature(slot, replica_to_temperature_[slot.replica_id]);
  initialize_thermostat_rng(
    slot,
    config_.resume,
    config_.resume ? restart_bdp_states_[slot.replica_id] : std::string(),
    config_.internal_seed,
    "REMD");
}

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
  gpu_scale_replica_velocity<<<(number_of_atoms - 1) / 128 + 1, 128, 0, slot.stream.get()>>>(
    number_of_atoms,
    factor,
    slot.atom.velocity_per_atom.data(),
    slot.atom.velocity_per_atom.data() + number_of_atoms,
    slot.atom.velocity_per_atom.data() + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

void REMD_Driver::attempt_exchanges(const long long global_step, FILE* exchange_file)
{
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (const auto& pair : config_.pairs(exchange_count_)) {
    const int low_label = pair.first, high_label = pair.second;
    const int statistic = stats_index(pair);
    const int low_replica = temperature_to_replica_[low_label];
    const int high_replica = temperature_to_replica_[high_label];
    Replica_Slot& low = *runtime_.slots()[low_replica];
    Replica_Slot& high = *runtime_.slots()[high_replica];
    const double beta_low = config_.betas[low_label % config_.num_temperatures()];
    const double beta_high = config_.betas[high_label % config_.num_temperatures()];
    const double exponent =
      config_.is_npt ? remd_log_ratio(
                         temperature_of(low_label),
                         temperature_of(high_label),
                         pressure_of(low_label) / PRESSURE_UNIT_CONVERSION,
                         pressure_of(high_label) / PRESSURE_UNIT_CONVERSION,
                         low.potential_energy,
                         high.potential_energy,
                         low.box.get_volume(),
                         high.box.get_volume())
                     : (low.potential_energy - high.potential_energy) * (beta_low - beta_high);
    if (!std::isfinite(exponent))
      PRINT_INPUT_ERROR("REMD encountered a non-finite exchange log acceptance ratio.");
    const double probability = exponent >= 0.0 ? 1.0 : std::exp(exponent);
    const bool accepted = exponent >= 0.0 || uniform(exchange_rng_) < probability;
    ++exchange_attempts_[statistic];
    exchange_probability_sum_[statistic] += probability;

    if (exchange_file && config_.is_npt) {
      fprintf(
        exchange_file,
        "%lld %c %d %d %d %d %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %.16g %d\n",
        global_step,
        pair.direction == 0 ? 'T' : 'P',
        low_replica,
        high_replica,
        low_label,
        high_label,
        temperature_of(low_label),
        temperature_of(high_label),
        pressure_of(low_label),
        pressure_of(high_label),
        low.potential_energy,
        high.potential_energy,
        low.box.get_volume(),
        high.box.get_volume(),
        exponent,
        probability,
        accepted ? 1 : 0);
    } else if (exchange_file) {
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

    ++exchange_accepts_[statistic];
    if (temperature_of(low_label) != temperature_of(high_label)) {
      scale_velocity(low, std::sqrt(temperature_of(high_label) / temperature_of(low_label)));
      scale_velocity(high, std::sqrt(temperature_of(low_label) / temperature_of(high_label)));
    }
    set_temperature(low, high_label);
    set_temperature(high, low_label);
    if (config_.is_npt && temperature_of(low_label) != temperature_of(high_label)) {
      // Include the rescaled velocities in a pending pressure measurement.
      for (Replica_Slot* slot : {&low, &high})
        slot->barostat->refresh_pending_pressure(
          slot->box, slot->atom, *slot->thermostat, slot->thermo, slot->stream);
    }
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
    "    Equilibrating all replicas independently for %d steps.\n", config_.equilibration_steps);
  fflush(stdout);
  const int progress_interval = std::max(1, config_.equilibration_steps / 10);
  int completed = 0;
  while (completed < config_.equilibration_steps) {
    const int steps = std::min(progress_interval, config_.equilibration_steps - completed);
    runtime_.advance_all(steps);
    completed += steps;
    printf("    %d REMD equilibration steps completed.\n", completed);
    fflush(stdout);
  }
}

void REMD_Driver::write_temperature_ladder() const
{
  if (config_.is_npt) {
    FILE* file = my_fopen("remd_npt_conditions.out", "w");
    fprintf(file, "# label temperature_index pressure_index temperature(K) pressure(GPa)\n");
    for (int label = 0; label < config_.replicas; ++label)
      fprintf(
        file,
        "%d %d %d %.16g %.16g\n",
        label,
        label % config_.num_temperatures(),
        label / config_.num_temperatures(),
        temperature_of(label),
        pressure_of(label));
    fclose(file);
    return;
  }
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
    if (config_.is_npt) {
      fprintf(
        file,
        " %d %.16g %.16g",
        slot->temperature_label,
        slot->potential_energy,
        slot->box.get_volume());
      continue;
    }
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
  if (config_.is_npt) {
    FILE* file = my_fopen("remd_npt_pairs.out", "w");
    fprintf(
      file,
      "# direction label_a label_b T_a T_b P_a P_b attempts accepted acceptance_ratio "
      "average_probability\n");
    for (int direction = 0; direction < 2; ++direction)
      for (int label = 0; label < config_.replicas; ++label) {
        const int t = label % config_.num_temperatures(), p = label / config_.num_temperatures();
        if (
          (direction == 0 && t + 1 >= config_.num_temperatures()) ||
          (direction == 1 && p + 1 >= config_.num_pressures))
          continue;
        const int other = label + (direction == 0 ? 1 : config_.num_temperatures());
        const int index = direction * config_.replicas + label;
        const long long attempts = exchange_attempts_[index];
        fprintf(
          file,
          "%c %d %d %.16g %.16g %.16g %.16g %lld %lld %.10e %.10e\n",
          direction == 0 ? 'T' : 'P',
          label,
          other,
          temperature_of(label),
          temperature_of(other),
          pressure_of(label),
          pressure_of(other),
          attempts,
          exchange_accepts_[index],
          attempts ? static_cast<double>(exchange_accepts_[index]) / attempts : 0,
          attempts ? exchange_probability_sum_[index] / attempts : 0);
      }
    fclose(file);
    return;
  }
  FILE* file = my_fopen("remd_ladder.out", "w");
  fprintf(file, "# REMD temperature ladder report\n");
  fprintf(file, "# target_acceptance %.10e\n", target_acceptance);
  fprintf(
    file,
    "# temp_i temp_j temperature_i temperature_j attempts accepted acceptance_ratio "
    "average_probability meets_target\n");
  bool all_meet_target = true;
  for (int label = 0; label + 1 < config_.replicas; ++label) {
    const long long attempts = exchange_attempts_[label];
    const double ratio =
      attempts > 0 ? static_cast<double>(exchange_accepts_[label]) / attempts : 0.0;
    const double average = attempts > 0 ? exchange_probability_sum_[label] / attempts : 0.0;
    const bool meets_target = attempts > 0 && ratio >= target_acceptance;
    all_meet_target = all_meet_target && meets_target;
    fprintf(
      file,
      "%d %d %.16g %.16g %lld %lld %.10e %.10e %d\n",
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
  if (extension != std::string::npos && (separator == std::string::npos || extension > separator))
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
      trajectory_filename(config_.dump_prefix, replica).c_str(), config_.resume ? "a" : "w");
}

void REMD_Driver::close_trajectory_files()
{
  for (FILE* file : trajectory_files_) {
    if (file)
      fclose(file);
  }
  trajectory_files_.clear();
}

void REMD_Driver::write_trajectories(const long long step)
{
  if (!config_.dump_xyz)
    return;
  const Replica_XYZ_Fields& quantities = config_.dump_fields;
  const bool velocity = config_.dump_velocity;
  const int n = source_atom_.number_of_atoms;
  const int tensor_order[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2}; // xx xy xz yx yy yz zx zy zz
  std::vector<double> force, potential, virial;
  for (Replica_Slot* slot : runtime_.slots()) {
    CHECK(gpuSetDevice(slot->device_id));
    slot->stream.synchronize();
    slot->atom.position_per_atom.copy_to_host(slot->atom.cpu_position_per_atom.data());
    if (velocity)
      slot->atom.velocity_per_atom.copy_to_host(slot->atom.cpu_velocity_per_atom.data());
    if (quantities.force) {
      force.resize(n * 3);
      slot->atom.force_per_atom.copy_to_host(force.data());
    }
    if (quantities.potential) {
      potential.resize(n);
      slot->atom.potential_per_atom.copy_to_host(potential.data());
    }
    if (quantities.virial) {
      virial.resize(n * 9);
      slot->atom.virial_per_atom.copy_to_host(virial.data());
    }
    FILE* file = trajectory_files_[slot->replica_id];
    fprintf(
      file,
      "%d\nTime=%.8f step=%lld replica=%d temperature_label=%d target_temperature=%.16g ",
      n,
      step * time_step_ * TIME_UNIT_CONVERSION,
      step,
      slot->replica_id,
      slot->temperature_label,
      slot->target_temperature);
    if (config_.is_npt)
      fprintf(
        file,
        "condition_label=%d target_pressure=%.16g potential_energy=%.16g ",
        slot->temperature_label,
        pressure_of(slot->temperature_label),
        slot->potential_energy);
    write_xyz_box(file, slot->box, 16);
    fprintf(
      file,
      "Properties=species:S:1:pos:R:3%s%s%s%s%s\n",
      quantities.mass ? ":mass:R:1" : "",
      velocity ? ":vel:R:3" : "",
      quantities.force ? ":forces:R:3" : "",
      quantities.potential ? ":energy_atom:R:1" : "",
      quantities.virial ? ":virial:R:9" : "");
    for (int i = 0; i < n; ++i) {
      fprintf(
        file,
        "%s %.16g %.16g %.16g",
        source_atom_.cpu_atom_symbol[i].c_str(),
        slot->atom.cpu_position_per_atom[i],
        slot->atom.cpu_position_per_atom[i + n],
        slot->atom.cpu_position_per_atom[i + 2 * n]);
      if (quantities.mass)
        fprintf(file, " %.16g", source_atom_.cpu_mass[i]);
      if (velocity)
        for (int d = 0; d < 3; ++d)
          fprintf(
            file, " %.16g", slot->atom.cpu_velocity_per_atom[i + d * n] / TIME_UNIT_CONVERSION);
      if (quantities.force)
        for (int d = 0; d < 3; ++d)
          fprintf(file, " %.16g", force[i + d * n]);
      if (quantities.potential)
        fprintf(file, " %.16g", potential[i]);
      if (quantities.virial)
        for (int d = 0; d < 9; ++d)
          fprintf(file, " %.16g", virial[i + tensor_order[d] * n]);
      fprintf(file, "\n");
    }
    fflush(file);
  }
}

void REMD_Driver::write_restart_xyz(Replica_Slot& slot) const
{
  char fields[512];
  snprintf(
    fields,
    sizeof(fields),
    "replica_mode=remd replica=%d temperature_label=%d target_temperature=%.17g "
    "potential_energy=%.17g restart_step=%lld",
    slot.replica_id,
    slot.temperature_label,
    slot.target_temperature,
    slot.potential_energy,
    completed_steps_ + number_of_steps_);
  if (config_.is_npt) {
    const size_t length = std::strlen(fields);
    snprintf(
      fields + length,
      sizeof(fields) - length,
      " ensemble=npt_scr condition_label=%d target_pressure=%.17g",
      slot.temperature_label,
      pressure_of(slot.temperature_label));
  }
  replica::write_restart_xyz(
    "remd_replica_" + std::to_string(slot.replica_id) + "_restart.xyz", slot, source_atom_, fields);
}

void REMD_Driver::write_restart_metadata() const
{
  FILE* file = my_fopen("remd_restart.meta", "w");
  fprintf(file, "# GPUMD REMD restart metadata\n");
  fprintf(
    file,
    "format_version %d\n",
    config_.is_npt ? npt_restart_format_version : restart_format_version);
  fprintf(file, "mode remd\n");
  if (config_.is_npt)
    fprintf(file, "ensemble npt_scr\n");
  fprintf(file, "replicas %d\n", config_.replicas);
  fprintf(file, "completed_steps %lld\n", completed_steps_ + number_of_steps_);
  fprintf(file, "exchange_count %lld\n", exchange_count_);
  fprintf(file, "exchange_interval %d\n", config_.exchange_interval);
  fprintf(file, "time_step %.17g\n", time_step_);
  fprintf(file, "temperature_coupling %.17g\n", temperature_coupling_);
  fprintf(file, "potential_hash %llu\n", potential_hash_);
  fprintf(file, "kspace %s\n", kspace_method_.c_str());
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
  for (const long long value : exchange_attempts_)
    fprintf(file, " %lld", value);
  fprintf(file, "\nexchange_accepts");
  for (const long long value : exchange_accepts_)
    fprintf(file, " %lld", value);
  fprintf(file, "\nexchange_probability_sum");
  for (const double value : exchange_probability_sum_)
    fprintf(file, " %.17g", value);
  std::ostringstream exchange_state;
  exchange_state << exchange_rng_;
  fprintf(file, "\nexchange_rng %s\n", exchange_state.str().c_str());
  if (config_.is_npt)
    write_npt_checkpoint(file);
  for (const Replica_Slot* slot : runtime_.slots()) {
    fprintf(
      file, "bdp_rng %d %s\n", slot->replica_id, slot->thermostat->export_rng_state().c_str());
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
  read_restart_metadata_header(
    input, "REMD", "remd", config_.is_npt ? npt_restart_format_version : restart_format_version);
  if (config_.is_npt && read_restart_line(input, "REMD", "ensemble") != "npt_scr")
    PRINT_INPUT_ERROR("REMD restart ensemble does not match the input.");

  int replicas = 0;
  read_restart_value(input, "REMD", "replicas", replicas);
  if (replicas != config_.replicas)
    PRINT_INPUT_ERROR("REMD restart replicas do not match the input.");
  read_restart_value(input, "REMD", "completed_steps", completed_steps_);
  if (completed_steps_ < 0)
    PRINT_INPUT_ERROR("REMD restart completed_steps is invalid.");
  read_restart_value(input, "REMD", "exchange_count", exchange_count_);
  if (exchange_count_ < 0)
    PRINT_INPUT_ERROR("REMD restart exchange_count is invalid.");
  int integer_value = 0;
  read_restart_value(input, "REMD", "exchange_interval", integer_value);
  if (integer_value != config_.exchange_interval)
    PRINT_INPUT_ERROR("REMD restart exchange_interval does not match the input.");
  double value = 0.0;
  read_restart_value(input, "REMD", "time_step", value);
  if (!values_match(value, time_step_))
    PRINT_INPUT_ERROR("REMD restart time_step does not match the input.");
  read_restart_value(input, "REMD", "temperature_coupling", value);
  if (!values_match(value, temperature_coupling_))
    PRINT_INPUT_ERROR("REMD restart nvt_bdp coupling does not match the input.");
  unsigned long long stored_hash = 0;
  read_restart_value(input, "REMD", "potential_hash", stored_hash);
  if (stored_hash != potential_hash_)
    PRINT_INPUT_ERROR("REMD restart potential file does not match the input.");
  // The method, followed by the PPPM spacing when it is given in run.in.
  if (read_restart_line(input, "REMD", "kspace") != kspace_method_)
    PRINT_INPUT_ERROR("REMD restart kspace does not match the input.");

  read_restart_key(input, "REMD", "temperatures");
  for (int r = 0; r < config_.num_temperatures(); ++r) {
    if (!(input >> value) || !values_match(value, config_.temperatures[r]))
      PRINT_INPUT_ERROR("REMD restart temperature ladder does not match the input.");
  }
  read_restart_key(input, "REMD", "replica_to_temperature");
  for (int& label : replica_to_temperature_) {
    if (!(input >> label))
      PRINT_INPUT_ERROR("REMD restart replica_to_temperature is incomplete.");
  }
  read_restart_key(input, "REMD", "temperature_to_replica");
  for (int& replica : temperature_to_replica_) {
    if (!(input >> replica))
      PRINT_INPUT_ERROR("REMD restart temperature_to_replica is incomplete.");
  }
  read_restart_key(input, "REMD", "exchange_attempts");
  for (auto& attempts : exchange_attempts_) {
    if (!(input >> attempts) || attempts < 0)
      PRINT_INPUT_ERROR("REMD restart exchange_attempts is invalid.");
  }
  read_restart_key(input, "REMD", "exchange_accepts");
  for (auto& accepts : exchange_accepts_) {
    if (!(input >> accepts) || accepts < 0)
      PRINT_INPUT_ERROR("REMD restart exchange_accepts is invalid.");
  }
  read_restart_key(input, "REMD", "exchange_probability_sum");
  for (double& sum : exchange_probability_sum_) {
    if (!(input >> sum) || !std::isfinite(sum) || sum < 0.0)
      PRINT_INPUT_ERROR("REMD restart exchange_probability_sum is invalid.");
  }
  std::istringstream exchange_state(read_restart_line(input, "REMD", "exchange_rng"));
  std::string trailing;
  if (!(exchange_state >> exchange_rng_) || exchange_state >> trailing)
    PRINT_INPUT_ERROR("REMD restart exchange_rng is invalid.");
  if (config_.is_npt)
    read_npt_checkpoint(input);
  read_restart_bdp_states(input, "REMD", config_.replicas, restart_bdp_states_);

  if (exchange_count_ != completed_steps_ / config_.exchange_interval)
    PRINT_INPUT_ERROR("REMD restart exchange_count is inconsistent with completed_steps.");
  const auto expected_attempts = config_.attempt_counts(exchange_count_);
  for (size_t i = 0; i < exchange_attempts_.size(); ++i) {
    if (
      exchange_attempts_[i] != expected_attempts[i] ||
      exchange_accepts_[i] > exchange_attempts_[i] ||
      exchange_probability_sum_[i] > exchange_attempts_[i] + 1.0e-12)
      PRINT_INPUT_ERROR("REMD restart exchange statistics are inconsistent.");
  }
  validate_restart_mappings();
}

void REMD_Driver::read_restart_xyz(Replica_Slot& slot)
{
  const std::string filename =
    config_.resume_prefix + "_replica_" + std::to_string(slot.replica_id) + "_restart.xyz";
  std::ifstream input(filename);
  const std::string comment = read_restart_xyz_header(
    input, filename, source_atom_, source_box_, "REMD", config_.is_npt ? &slot.box : nullptr);
  std::string value;
  int file_replica = -1;
  int temperature_label = -1;
  long long restart_step = -1;
  double target_temperature = 0.0;
  if (
    !extract_token_value(comment, "replica_mode", value) || value != "remd" ||
    !extract_token_value(comment, "replica", value) || !parse_scalar(value, file_replica) ||
    file_replica != slot.replica_id || !extract_token_value(comment, "temperature_label", value) ||
    !parse_scalar(value, temperature_label) ||
    temperature_label != replica_to_temperature_[slot.replica_id] ||
    !extract_token_value(comment, "target_temperature", value) ||
    !parse_scalar(value, target_temperature) ||
    !values_match(target_temperature, temperature_of(temperature_label)) ||
    !extract_token_value(comment, "potential_energy", value) ||
    !parse_scalar(value, slot.potential_energy) || !std::isfinite(slot.potential_energy) ||
    !extract_token_value(comment, "restart_step", value) || !parse_scalar(value, restart_step) ||
    restart_step != completed_steps_)
    PRINT_INPUT_ERROR("REMD restart xyz metadata does not match the restart mapping.");
  if (config_.is_npt) {
    double pressure = 0;
    int label = -1;
    if (
      !extract_token_value(comment, "ensemble", value) || value != "npt_scr" ||
      !extract_token_value(comment, "condition_label", value) || !parse_scalar(value, label) ||
      label != temperature_label || !extract_token_value(comment, "target_pressure", value) ||
      !parse_scalar(value, pressure) || !values_match(pressure, pressure_of(label)))
      PRINT_INPUT_ERROR("REMD restart xyz NPT conditions do not match the metadata.");
  }
  read_restart_xyz_atoms(input, comment, source_atom_, slot.atom, "REMD");
}

void REMD_Driver::write_npt_checkpoint(FILE* file) const
{
  fprintf(file, "grid %d %d\npressure_table", config_.num_temperatures(), config_.num_pressures);
  for (double pressure : config_.pressure_table)
    fprintf(file, " %.17g", pressure);
  fprintf(
    file,
    "\nbarostat %.17g %.17g %d\n",
    barostat_input_.bulk_modulus,
    barostat_input_.pressure_coupling,
    config_.pcouple_interval);
  for (const Replica_Slot* slot : runtime_.slots())
    fprintf(file, "scr_state %d %s\n", slot->replica_id, slot->barostat->export_state().c_str());
}

void REMD_Driver::read_npt_checkpoint(std::istream& input)
{
  int nt = 0, np = 0;
  read_restart_value(input, "REMD", "grid", nt);
  if (!(input >> np) || nt != config_.num_temperatures() || np != config_.num_pressures)
    PRINT_INPUT_ERROR("REMD restart grid does not match the input.");
  read_restart_key(input, "REMD", "pressure_table");
  for (double expected : config_.pressure_table) {
    double value = 0;
    if (!(input >> value) || !values_match(value, expected))
      PRINT_INPUT_ERROR("REMD restart pressure conditions do not match the input.");
  }
  double modulus = 0, tau = 0;
  int interval = 0;
  read_restart_key(input, "REMD", "barostat");
  if (
    !(input >> modulus >> tau >> interval) ||
    !values_match(modulus, barostat_input_.bulk_modulus) ||
    !values_match(tau, barostat_input_.pressure_coupling) || interval != config_.pcouple_interval)
    PRINT_INPUT_ERROR("REMD restart barostat does not match the input.");
  for (int r = 0; r < config_.replicas; ++r) {
    int id = -1;
    read_restart_value(input, "REMD", "scr_state", id);
    if (id != r)
      PRINT_INPUT_ERROR("REMD restart SCR replica order is invalid.");
    std::getline(input, restart_scr_states_[r]);
    restart_scr_states_[r] = trim_left(restart_scr_states_[r]);
  }
}

void REMD_Driver::run()
{
  validate_input();
  const auto initialization_begin = std::chrono::high_resolution_clock::now();
  runtime_.initialize(config_.replicas, config_.replicas_per_gpu, [this](Replica_Slot& slot) {
    initialize_slot(slot);
  });
  runtime_.start_workers();
  if (!config_.resume)
    equilibrate();
  const auto initialization_end = std::chrono::high_resolution_clock::now();

  write_temperature_ladder();
  bool new_exchange = false;
  bool new_status = false;
  bool new_thermo = false;
  FILE* exchange_file =
    open_output_file(output_file("exchange"), config_.resume, exchange_marker(), new_exchange);
  FILE* status_file =
    open_output_file(output_file("status"), config_.resume, status_marker(), new_status);
  FILE* thermo_file =
    open_output_file(output_file("thermo"), config_.resume, thermo_marker(), new_thermo);
  if (new_exchange && config_.is_npt) {
    fprintf(
      exchange_file,
      "# States before swapping; temperatures K, pressures GPa, energies eV, volumes A^3.\n");
    fprintf(
      exchange_file,
      "# step direction replica_i replica_j label_a label_b T_a T_b P_a P_b E_i E_j V_i V_j "
      "log_acceptance_ratio acceptance_probability accepted\n");
  } else if (new_exchange) {
    fprintf(exchange_file, "# Exchange states are recorded immediately before swapping.\n");
    fprintf(exchange_file, "# Units: target_temperature K, potential_energy eV.\n");
    fprintf(
      exchange_file,
      "# step replica_i replica_j temperature_label_i temperature_label_j "
      "target_temperature_i target_temperature_j potential_energy_i "
      "potential_energy_j log_acceptance_ratio acceptance_probability accepted\n");
  }
  if (new_status && config_.is_npt) {
    fprintf(status_file, "# step condition_label_of_replica_0 ...\n");
  } else if (new_status) {
    fprintf(
      status_file, "# Replica-to-temperature mapping after any exchange at the reported step.\n");
    fprintf(status_file, "# step");
    for (int replica = 0; replica < config_.replicas; ++replica)
      fprintf(status_file, " temperature_label_of_replica_%d", replica);
    fprintf(status_file, "\n");
  }
  if (new_thermo && config_.is_npt) {
    fprintf(
      thermo_file,
      "# States after exchange; step then condition_label energy(eV) volume(A^3) per replica.\n");
  } else if (new_thermo) {
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
  if (new_status)
    write_temperature_status(completed_steps_, status_file);
  if (new_thermo)
    write_thermo(completed_steps_, thermo_file);

  const auto production_begin = std::chrono::high_resolution_clock::now();
  const int progress_interval = std::max(1, number_of_steps_ / 10);
  int local_step = 0;
  while (local_step < number_of_steps_) {
    const long long global_step = completed_steps_ + local_step;
    int next_event =
      next_periodic_event(local_step, number_of_steps_, global_step, config_.exchange_interval);
    if (config_.dump_xyz) {
      next_event = std::min(
        next_event,
        next_periodic_event(local_step, number_of_steps_, global_step, config_.dump_interval));
    }
    const int next_progress =
      std::min(number_of_steps_, ((local_step / progress_interval) + 1) * progress_interval);
    next_event = std::min(next_event, next_progress);
    runtime_.advance_all(next_event - local_step);
    local_step = next_event;
    const long long current_global_step = completed_steps_ + local_step;

    const bool exchange_due = current_global_step % config_.exchange_interval == 0;
    const bool dump_due = config_.dump_xyz && current_global_step % config_.dump_interval == 0;
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

  runtime_.stop_workers();
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
  const Replica_Input& input,
  const RunInput& run_input,
  Atom& source_atom,
  Box& source_box,
  std::vector<Group>& source_group)
{
  REMD_Driver driver(input, run_input, source_atom, source_box, source_group);
  driver.run();
}
