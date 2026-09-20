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
#include "action.cuh"
#include "enhanced_sampling_engine.cuh"
#include <cstdio>
#include <string>
#include <vector>

class EnhancedSamplingAction : public Action
{
public:
  EnhancedSamplingAction(const std::vector<std::string>& tokens);
  ~EnhancedSamplingAction() override;

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void setup_force(
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void post_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

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
    Force& force) override;

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  void parse_input_file();
  void parse_output(const std::vector<std::string>& tokens);
  void parse_start(const std::vector<std::string>& tokens);
  void parse_cv(const std::vector<std::string>& tokens);
  void parse_bias(const std::vector<std::string>& tokens);
  bool should_output(const int step) const;
  void close_output();

  std::string input_filename_;
  std::string output_filename_;
  int input_line_;
  int output_interval_;
  bool output_is_set_;
  bool start_is_set_;
  FILE* output_;
  EnhancedSamplingEngine engine_;
};
