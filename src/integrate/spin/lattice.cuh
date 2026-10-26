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

#pragma once
#include "integrate/ensemble_nhc.cuh"
#include "integrate/ensemble_mttk.cuh"
#include <cstdint>

// Only Spin ensembles persist lattice baths and the original ramp protocol.
// The upstream base classes own and advance the mechanical state.
class Spin_Lattice_NHC final : public Ensemble_NHC
{
public:
  using Ensemble_NHC::Ensemble_NHC;
  void initialize_run(double time_step, Atom& atom, Box& box, const std::vector<Group>& group) override;
  void compute1(double time_step, int step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo) override;
  void compute2(double time_step, int step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo, Force& force) override;
  std::string restart_state() const;
  double protocol_fraction(int step, int number_of_steps) const;

private:
  void restore_restart(const std::string& state);
  std::vector<double> restart_configuration_;
  std::uint64_t restart_offset_ = 0, completed_steps_ = 0;
  int protocol_steps_ = 0;
};

class Spin_Lattice_MTTK final : public Ensemble_MTTK
{
public:
  using Ensemble_MTTK::Ensemble_MTTK;
  void initialize_before_first_step(double time_step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo) override;
  void compute2(double time_step, int step, int number_of_steps, const std::vector<Group>& group,
    Box& box, Atom& atom, GPU_Vector<double>& thermo, Force& force) override;
  std::string restart_state() const;
  double protocol_fraction(int step, int number_of_steps) const;
  double get_temperature_period() const { return t_period; }

private:
  void get_target_temp(int step, int number_of_steps, const std::vector<Group>& group,
    const Box& box, const Atom& atom, GPU_Vector<double>& thermo) override;
  void get_target_pressure(int step, int number_of_steps, const std::vector<Group>& group,
    const Box& box, const Atom& atom, GPU_Vector<double>& thermo) override;
  std::vector<double> restart_configuration() const;
  void restore_restart(const std::string& state);
  std::uint64_t restart_offset = 0, completed_steps = 0;
  int protocol_steps = 0;
  bool initialized = false;
};
