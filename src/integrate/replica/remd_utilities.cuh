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
Host-only exchange geometry, acceptance ratio, and SCR arithmetic of replica-
exchange molecular dynamics, shared with the CPU tests.

Major contributors:
  Jiuyang Shi: Temperature-pressure grid, exchange rule, and SCR update.
  Wenhao Luo: Shared host utilities for the REMD driver and its CPU tests.
------------------------------------------------------------------------------*/

#pragma once

#include "utilities/common.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace replica
{

struct REMD_Pair {
  int direction; // 0: temperature, 1: pressure
  int first;
  int second;
};

struct REMD_Grid {
  bool is_npt = false;
  int num_pressures = 1;
  std::vector<double> temperatures;
  std::vector<double> pressure_table; // GPa; row-major [temperature][pressure]

  int num_temperatures() const { return static_cast<int>(temperatures.size()); }
  int label(int t, int p) const { return p * num_temperatures() + t; }
  double temperature(int label) const { return temperatures[label % num_temperatures()]; }
  double pressure(int label) const
  {
    return is_npt ? pressure_table
                      [(label % num_temperatures()) * num_pressures + label / num_temperatures()]
                  : 0.0;
  }

  std::string validate(int replicas) const
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

inline double remd_log_ratio(
  double temperature_a,
  double temperature_b,
  double pressure_a,
  double pressure_b,
  double energy_i,
  double energy_j,
  double volume_i,
  double volume_j)
{
  const double beta_a = 1.0 / (K_B * temperature_a);
  const double beta_b = 1.0 / (K_B * temperature_b);
  return (beta_a - beta_b) * (energy_i - energy_j) +
         (beta_a * pressure_a - beta_b * pressure_b) * (volume_i - volume_j);
}

inline double scr_log_volume_increment(
  double temperature,
  double volume,
  double target_pressure,
  double pressure,
  double compressibility,
  double dt_over_tau,
  double gaussian)
{
  const double kT = K_B * temperature;
  return -compressibility * dt_over_tau * (target_pressure - pressure) +
         std::sqrt(2.0 * kT * compressibility * dt_over_tau / volume) * gaussian;
}

inline std::string export_scr_state(
  long long counter,
  bool pending,
  const double* thermo,
  const std::mt19937& rng,
  const std::normal_distribution<double>& normal)
{
  std::ostringstream out;
  out << 1 << ' ' << counter << ' ' << (pending ? 1 : 0) << ' ' << std::setprecision(17);
  for (int i = 0; i < 5; ++i)
    out << thermo[i] << ' ';
  out << rng << ' ' << normal;
  return out.str();
}

inline bool import_scr_state(
  const std::string& text,
  int interval,
  long long& counter,
  bool& pending,
  double* thermo,
  std::mt19937& rng,
  std::normal_distribution<double>& normal)
{
  std::istringstream in(text);
  int version = 0, flag = 0;
  long long step = 0;
  std::array<double, 5> values;
  std::mt19937 restored_rng;
  std::normal_distribution<double> restored_normal;
  if (
    interval <= 0 || !(in >> version >> step >> flag) || version != 1 || step < 0 ||
    (flag != 0 && flag != 1) || (flag && step % interval != 0))
    return false;
  for (double& value : values)
    if (!(in >> value) || !std::isfinite(value))
      return false;
  if (
    !(in >> restored_rng >> restored_normal) || restored_normal.mean() != 0.0 ||
    restored_normal.stddev() != 1.0)
    return false;
  in >> std::ws;
  if (!in.eof())
    return false;
  counter = step;
  pending = flag != 0;
  std::copy(values.begin(), values.end(), thermo);
  rng = restored_rng;
  normal = restored_normal;
  return true;
}

} // namespace replica
