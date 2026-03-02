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

#include "utilities/gpu_vector.cuh"
#include <vector>
#include <cstdio>

class Atom;
class Group;

#ifndef MAX_SPRING_CALLS
#define MAX_SPRING_CALLS 10
#endif

class Add_Spring
{
public:
  void parse(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom);
  void compute(const int step, const std::vector<Group>& groups, Atom& atom);
  void finalize();

private:
  int num_calls_ = 0;

  // spring mode for each call
  enum SpringMode {
    MODE_GHOST_COM  = 0,
    MODE_GHOST_ATOM = 1,
    MODE_COUPLE_COM = 2
  };

  enum StiffnessMode {
    STIFFNESS_COUPLE   = 0,
    STIFFNESS_DECOUPLE = 1
  };

  // per-call configuration
  SpringMode mode_[MAX_SPRING_CALLS];
  StiffnessMode stiffness_mode_[MAX_SPRING_CALLS];

  // group info
  int grouping_method_[MAX_SPRING_CALLS];  // grouping method for this call
  int group_id_[MAX_SPRING_CALLS];         // group id in the grouping method
  int group_id_2_[MAX_SPRING_CALLS];       // for couple_com: second group id from same method

  // spring parameters (per-call)
  double k_couple_[MAX_SPRING_CALLS];      // spring constant for couple mode
  double R0_[MAX_SPRING_CALLS];            // reference distance for couple mode
  double k_decouple_[MAX_SPRING_CALLS][3]; // spring constants (kx, ky, kz) for decouple mode

  // ghost motion parameters
  double velocity_[MAX_SPRING_CALLS][3];   // (vx, vy, vz) in Å/step
  double origin_[MAX_SPRING_CALLS][3];     // absolute initial position R_g(0), set at first compute
  double offset_[MAX_SPRING_CALLS][3];     // (x0, y0, z0) offset relative to initial COM
  int init_origin_[MAX_SPRING_CALLS];      // flag: 0 = not set, 1 = set

  // ghost_atom mode: per-atom anchor positions (allocated during parse)
  GPU_Vector<double> ghost_atom_pos_[MAX_SPRING_CALLS]; // device memory, size = 3 * group_size
  int ghost_atom_group_size_[MAX_SPRING_CALLS];

  // temporary device buffers
  GPU_Vector<double> d_tmp_vec3_;   // length 3 (for sum_mx, sum_my, sum_mz)
  GPU_Vector<double> d_tmp_scalar_; // length 1 (for sum_m or energy)
  GPU_Vector<double> d_tmp_force3_; // length 3 (for sum of spring forces)

  // per-call output (on host)
  double energy_[MAX_SPRING_CALLS];
  double force_[MAX_SPRING_CALLS][3];
  double total_force_[MAX_SPRING_CALLS];

  // output control
  FILE* fp_out_[MAX_SPRING_CALLS] = {nullptr};
  int output_stride_ = 100;
};
