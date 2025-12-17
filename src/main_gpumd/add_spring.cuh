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

#include <vector>
#include <cstdio>

class Atom;
class Group;


#ifndef MAX_SPRING_CALLS
#define MAX_SPRING_CALLS 10
#endif

enum SpringStiffMode {
  SPRING_COUPLE   = 0,
  SPRING_DECOUPLE = 1
};

class Add_Spring
{
public:
  void parse(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom);
  void compute(const int step, const std::vector<Group>& groups, Atom& atom);
  void finalize();

private:
  int num_calls_ = 0;

  // group info (one group per call)
  int grouping_method_[MAX_SPRING_CALLS];
  int group_id_[MAX_SPRING_CALLS];

  // motion: R_g(step) = R_g(0) + v * step
  double ghost_velocity_[MAX_SPRING_CALLS][3]; // (vx, vy, vz) in Å/step
  double ghost_origin_[MAX_SPRING_CALLS][3];   // absolute R_g(0), set at first compute
  double ghost_offset_[MAX_SPRING_CALLS][3];   // (x0,y0,z0) relative to initial COM
  int    init_origin_[MAX_SPRING_CALLS];       // 0 -> not set, 1 -> set

  // stiffness mode + parameters
  SpringStiffMode stiff_mode_[MAX_SPRING_CALLS];
  double k_couple_[MAX_SPRING_CALLS];
  double R0_[MAX_SPRING_CALLS];
  double k_decouple_[MAX_SPRING_CALLS][3]; // (kx, ky, kz)

  // temp buffers (device)
  double* d_tmp_vec3_   = nullptr; // length 3
  double* d_tmp_scalar_ = nullptr; // length 1

  // per-call outputs (host side)
  double spring_energy_[MAX_SPRING_CALLS];
  double spring_force_[MAX_SPRING_CALLS][3];
  double spring_fric_[MAX_SPRING_CALLS];

  // output control
  FILE* fp_out_[MAX_SPRING_CALLS] = {nullptr}; // output file pointers
  int   output_stride_ = 100; // write every N steps (set to 0 to disable)
  int   printed_use_wrapped_position_ = 0;
};
