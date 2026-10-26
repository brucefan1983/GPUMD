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

#include "lattice.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

void Spin_Lattice_NHC::initialize_run(
  double time_step, Atom& atom, Box& box, const std::vector<Group>& group)
{
  Ensemble_NHC::initialize_run(time_step, atom, box, group);
  restart_configuration_ = {double(NOSE_HOOVER_CHAIN_LENGTH), double(atom.number_of_atoms),
    time_step, get_temperature1(), get_temperature2(), temperature_coupling,
    double(fixed_group), double(move_group), double(fixed_grouping_method), double(move_grouping_method)};
  for (int i = 0; i < 3; ++i) restart_configuration_.push_back(move_group >= 0 ? move_velocity[i] : 0.0);
  if (!atom.spin_lattice_restart.nhc.empty()) {
    restore_restart(atom.spin_lattice_restart.nhc);
    atom.spin_lattice_restart.nhc.clear();
    printf("Restored NVT NHC endpoint after %llu steps (protocol length %d).\n",
      static_cast<unsigned long long>(restart_offset_), protocol_steps_);
  }
}

void Spin_Lattice_NHC::compute1(double time_step, int step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  if (protocol_steps_ == 0) protocol_steps_ = number_of_steps;
  Ensemble_NHC::compute1(time_step, step, number_of_steps, group, box, atom, thermo);
}

void Spin_Lattice_NHC::compute2(double time_step, int step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo, Force& force)
{
  Ensemble_NHC::compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
  completed_steps_ = restart_offset_ + static_cast<std::uint64_t>(step) + 1;
}

double Spin_Lattice_NHC::protocol_fraction(int step, int number_of_steps) const
{
  const int length = protocol_steps_ ? protocol_steps_ : number_of_steps;
  return std::min(1.0, double(restart_offset_ + step) / length);
}

std::string Spin_Lattice_NHC::restart_state() const
{
  if (type != EnsembleType::NVT_NHC || protocol_steps_ == 0) return "";
  std::vector<double> values = {1.0, double(completed_steps_), double(protocol_steps_)};
  values.insert(values.end(), restart_configuration_.begin(), restart_configuration_.end());
  for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; ++i) {
    // vel_nhc1 stores conjugate momentum: nhc() advances pos by dt * vel / mass.
    values.insert(values.end(), {mas_nhc1[i], pos_nhc1[i], vel_nhc1[i]});
  }
  std::ostringstream stream;
  stream << std::setprecision(17);
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) PRINT_INPUT_ERROR("Cannot write non-finite NHC checkpoint.");
    if (i) stream << ',';
    stream << values[i];
  }
  return stream.str();
}

void Spin_Lattice_NHC::restore_restart(const std::string& state)
{
  std::vector<double> values;
  std::istringstream input(state);
  std::string token;
  if (state.empty() || state.back() == ',') PRINT_INPUT_ERROR("Invalid NHC checkpoint.");
  while (std::getline(input, token, ',')) {
    try {
      size_t end = 0;
      const double value = std::stod(token, &end);
      if (end != token.size() || !std::isfinite(value)) throw std::invalid_argument("value");
      values.push_back(value);
    } catch (...) { PRINT_INPUT_ERROR("Invalid number in NHC checkpoint."); }
  }
  const size_t expected = 3 + restart_configuration_.size() + 3 * NOSE_HOOVER_CHAIN_LENGTH;
  if (values.size() != expected || values[0] != 1.0) PRINT_INPUT_ERROR("Unsupported or truncated NHC checkpoint.");
  if (values[1] < 0 || values[1] > 9007197107257343.0 || std::floor(values[1]) != values[1] ||
      values[2] < 1 || values[2] > 2147483647.0 || std::floor(values[2]) != values[2]) {
    PRINT_INPUT_ERROR("Invalid NHC checkpoint step/protocol length.");
  }
  for (size_t i = 0; i < restart_configuration_.size(); ++i) {
    if (values[3 + i] != restart_configuration_[i]) {
      PRINT_INPUT_ERROR("NHC restart parameters, atom count or timestep do not match.");
    }
  }
  restart_offset_ = completed_steps_ = static_cast<std::uint64_t>(values[1]);
  protocol_steps_ = static_cast<int>(values[2]);
  size_t index = 3 + restart_configuration_.size();
  for (int i = 0; i < NOSE_HOOVER_CHAIN_LENGTH; ++i) {
    mas_nhc1[i] = values[index++]; pos_nhc1[i] = values[index++]; vel_nhc1[i] = values[index++];
    if (mas_nhc1[i] <= 0.0) PRINT_INPUT_ERROR("NHC checkpoint masses must be positive.");
  }
}

void Spin_Lattice_MTTK::initialize_before_first_step(double time_step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo)
{
  Ensemble_MTTK::initialize_before_first_step(time_step, number_of_steps, group, box, atom, thermo);
  protocol_steps = number_of_steps;
  if (!atom.spin_lattice_restart.mttk.empty()) {
    restore_restart(atom.spin_lattice_restart.mttk);
    atom.spin_lattice_restart.mttk.clear();
    printf("Restored MTTK endpoint after %llu steps (protocol length %d).\n",
      static_cast<unsigned long long>(restart_offset), protocol_steps);
  }
  initialized = true;
}

void Spin_Lattice_MTTK::compute2(double time_step, int step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo, Force& force)
{
  Ensemble_MTTK::compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
  completed_steps = restart_offset + static_cast<std::uint64_t>(step) + 1;
}

void Spin_Lattice_MTTK::get_target_temp(
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  t_target = t_start + (t_stop - t_start) * protocol_fraction(step, number_of_steps);
}

void Spin_Lattice_MTTK::get_target_pressure(
  const int step,
  const int number_of_steps,
  const std::vector<Group>& group,
  const Box& box,
  const Atom& atom,
  GPU_Vector<double>& thermo)
{
  double delta = protocol_fraction(step, number_of_steps);
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 3; y++) {
      p_target[x][y] = p_start[x][y] + (p_stop[x][y] - p_start[x][y]) * delta;
    }
  }
  get_p_hydro();
  if (non_hydrostatic) {
    // get_sigma uses its step only for the reference-cell reset test. Reduce
    // the absolute checkpoint counter before narrowing to the upstream int.
    const int reset_step = h0_reset_interval > 0 ?
      static_cast<int>((restart_offset + step) % h0_reset_interval) : step;
    get_sigma(reset_step, box);
  }
}

// Only persistent endpoint variables are serialized. Matrix scratch buffers,
// frequencies and substep sizes are recomputed from the checked configuration.
std::vector<double> Spin_Lattice_MTTK::restart_configuration() const
{
  std::vector<double> values = {
    double(ensemble_type), double(use_thermostat), double(use_barostat), double(couple_type),
    double(non_hydrostatic), double(h0_reset_interval), double(tchain), double(pchain),
    double(temperature_dof), dt, t_start, t_stop, t_period,
    double(fixed_group), double(move_group), double(fixed_grouping_method), double(move_grouping_method)};
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
    values.insert(values.end(), {p_start[i][j], p_stop[i][j], p_period[i][j],
      double(p_flag[i][j]), double(need_scale[i][j])});
  }
  return values;
}

double Spin_Lattice_MTTK::protocol_fraction(int step, int number_of_steps) const
{
  const int length = protocol_steps ? protocol_steps : number_of_steps;
  return std::min(1.0, double(restart_offset + step) / length);
}

std::string Spin_Lattice_MTTK::restart_state() const
{
  if (!initialized) return "";
  // Numeric CSV avoids whitespace/quote ambiguity in extended XYZ headers.
  std::vector<double> values = {1.0, double(completed_steps), double(protocol_steps)};
  const auto config = restart_configuration();
  values.insert(values.end(), config.begin(), config.end());
  values.insert(values.end(), {t_current, t_target, t_for_barostat,
    double(t_current_from_thermo), factor_eta, vol_ref});
  for (auto matrix : {h_ref_inv, omega_dot, omega_mass, p_current, p_target, p_hydro, sigma}) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) values.push_back(matrix[i][j]);
  }
  for (int n = 0; n < tchain; ++n) values.insert(values.end(), {Q[n], eta_dot[n], eta_dotdot[n]});
  values.push_back(eta_dot[tchain]);
  for (int n = 0; n < pchain; ++n) values.insert(values.end(), {Q_p[n], eta_p_dot[n], eta_p_dotdot[n]});
  values.push_back(eta_p_dot[pchain]);
  std::ostringstream stream;
  stream << std::setprecision(17);
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i])) PRINT_INPUT_ERROR("Cannot write non-finite MTTK checkpoint.");
    if (i) stream << ',';
    stream << values[i];
  }
  return stream.str();
}

void Spin_Lattice_MTTK::restore_restart(const std::string& state)
{
  std::vector<double> values;
  std::istringstream input(state);
  std::string token;
  if (state.empty() || state.back() == ',') PRINT_INPUT_ERROR("Invalid MTTK checkpoint.");
  while (std::getline(input, token, ',')) {
    try {
      size_t end = 0;
      double value = std::stod(token, &end);
      if (end != token.size() || !std::isfinite(value)) throw std::invalid_argument("value");
      values.push_back(value);
    } catch (...) { PRINT_INPUT_ERROR("Invalid number in MTTK checkpoint."); }
  }
  const auto config = restart_configuration();
  const size_t expected = 3 + config.size() + 6 + 7 * 9 + 3 * (tchain + pchain) + 2;
  if (values.size() != expected || values[0] != 1.0) PRINT_INPUT_ERROR("Unsupported or truncated MTTK checkpoint.");
  if (values[1] < 0 || values[1] > 9007197107257343.0 || std::floor(values[1]) != values[1] ||
      values[2] < 1 || values[2] > 2147483647.0 || std::floor(values[2]) != values[2]) {
    PRINT_INPUT_ERROR("Invalid MTTK checkpoint step/protocol length.");
  }
  for (size_t i = 0; i < config.size(); ++i) {
    if (values[3 + i] != config[i]) PRINT_INPUT_ERROR("MTTK restart parameters, atom count or timestep do not match.");
  }
  restart_offset = completed_steps = static_cast<std::uint64_t>(values[1]);
  protocol_steps = static_cast<int>(values[2]);
  size_t index = 3 + config.size();
  t_current = values[index++]; t_target = values[index++]; t_for_barostat = values[index++];
  if (values[index] != 0.0 && values[index] != 1.0) PRINT_INPUT_ERROR("Invalid MTTK temperature-cache flag.");
  t_current_from_thermo = values[index++] != 0;
  factor_eta = values[index++]; vol_ref = values[index++];
  for (auto matrix : {h_ref_inv, omega_dot, omega_mass, p_current, p_target, p_hydro, sigma}) {
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) matrix[i][j] = values[index++];
  }
  for (int n = 0; n < tchain; ++n) { Q[n] = values[index++]; eta_dot[n] = values[index++]; eta_dotdot[n] = values[index++]; }
  eta_dot[tchain] = values[index++];
  for (int n = 0; n < pchain; ++n) { Q_p[n] = values[index++]; eta_p_dot[n] = values[index++]; eta_p_dotdot[n] = values[index++]; }
  eta_p_dot[pchain] = values[index++];
}
