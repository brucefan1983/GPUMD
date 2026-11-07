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
#include "action.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
#include <string>
#include <vector>

class Atom;
class Box;
class Force;
class Group;
class Integrate;
class Ensemble;
class Measure
{
public:
  bool has_actions() const { return !actions_.empty(); }
  // Rerun-specific dispatch reuses the existing measurement Actions, without
  // invoking their integration/force modification hooks.
  void validate_rerun(bool has_potential) const;
  bool rerun_needs_thermo() const;
  bool rerun_needs_velocity() const;
  bool rerun_output_conflicts(const std::string& trajectory) const;
  void set_xyz_header_step(long long timestep);

  void pre_rerun(
    const double time_step, const double temperature, Integrate& integrate, std::vector<Group>& group,
    Atom& atom, Box& box, Force& force, const std::string& trajectory);
  void rerun_frame(
    const long long timestep, const double time_natural,
    Integrate& integrate, std::vector<Group>& group, Atom& atom,
    Box& box, Force& force, GPU_Vector<double>& thermo);
  void post_rerun(
    const double time_step, Integrate& integrate, Atom& atom, Box& box);


  bool parse_action(
    const std::vector<std::string>& tokens,
    const int number_of_types,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force,
    const std::string& first_potential_filename);

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  void setup_force(
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

  void end_of_step(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force);

  void post_integrate1(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  void pre_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  void post_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

private:
  std::vector<std::unique_ptr<Action>> actions_;
};
