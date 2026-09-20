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
#include "ensemble.cuh"
#include <string>

class Ensemble_NHC : public Ensemble
{
public:
  Ensemble_NHC(
    const std::vector<std::string>& tokens, const std::vector<Group>& group);

  double get_temperature1() const;
  double get_temperature2() const;

  void initialize_run(
    const double time_step, Atom& atom, Box& box, const std::vector<Group>& group) override;

  void compute1(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo) override;

  void compute2(
    const double time_step,
    const int step,
    const int number_of_steps,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo,
    Force& force) override;

protected:
  double mas_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
  double pos_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
  double vel_nhc1[NOSE_HOOVER_CHAIN_LENGTH];
  double mas_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
  double pos_nhc2[NOSE_HOOVER_CHAIN_LENGTH];
  double vel_nhc2[NOSE_HOOVER_CHAIN_LENGTH];

  void integrate_nvt_nhc_1(
    const double time_step,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo);

  void integrate_nvt_nhc_2(
    const double time_step,
    const double volume,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& potential_per_atom,
    const GPU_Vector<double>& force_per_atom,
    const GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& thermo);

  void integrate_heat_nhc_1(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void integrate_heat_nhc_2(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void integrate_heat_nhc_power_1(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

  void integrate_heat_nhc_power_2(
    const double time_step,
    const std::vector<Group>& group,
    const GPU_Vector<double>& mass,
    const GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& velocity_per_atom);

private:
  void parse(
    const std::vector<std::string>& tokens, const std::vector<Group>& group);
  void parse_heat_groups(
    const std::vector<std::string>& tokens, const std::vector<Group>& group);

  double temperature1_ = 0.0;
  double temperature2_ = 0.0;
};
