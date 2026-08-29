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
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <vector>

class Add_Spring : public Action
{
public:
  Add_Spring(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom);
  ~Add_Spring() override;

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

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  enum SpringMode {
    MODE_GHOST_COM = 0,
    MODE_GHOST_ATOM = 1,
    MODE_COM_COM = 2
  };

  enum StiffnessMode {
    STIFFNESS_COUPLE = 0,
    STIFFNESS_DECOUPLE = 1
  };

  void save_restart();
  void load_restart();
  void advance_ghost();
  void apply_force(std::vector<Group>& groups, Atom& atom);
  void write_output(const int step);

  SpringMode mode_ = MODE_GHOST_COM;
  StiffnessMode stiffness_mode_ = STIFFNESS_COUPLE;

  int grouping_method_ = 0;
  int group_id_ = 0;
  int group_id_2_ = 0;
  int spring_id_ = -1;
  int continue_spring_id_ = -1;
  int restart_group_size_ = 0;

  double k_couple_ = 0.0;
  double R0_ = 0.0;
  double k_decouple_[3] = {0.0, 0.0, 0.0};
  double velocity_[3] = {0.0, 0.0, 0.0};
  double origin_[3] = {0.0, 0.0, 0.0};
  double offset_[3] = {0.0, 0.0, 0.0};
  bool init_origin_ = false;

  GPU_Vector<double> ghost_atom_pos_;
  int ghost_atom_group_size_ = 0;

  GPU_Vector<double> d_tmp_vec3_;
  GPU_Vector<double> d_tmp_scalar_;
  GPU_Vector<double> d_tmp_force3_;

  double energy_ = 0.0;
  double force_[3] = {0.0, 0.0, 0.0};
  double total_force_ = 0.0;

  FILE* fp_out_ = nullptr;
  int output_stride_ = 100;
};
