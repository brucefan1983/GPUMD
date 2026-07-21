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

/*----------------------------------------------------------------------------80
The Deposition class splits a run.in file into sub-run input files according to
the deposition keyword and performs atom insertion between consecutive sub-runs.
------------------------------------------------------------------------------*/

#pragma once
#include <random>
#include <string>
#include <vector>

class Deposition {
public:
  std::vector<std::vector<std::string>> subrun_lines;

  int interval = 0;
  int direction = 2;
  int num_subruns = 0;
  double height_min = 0.0;
  double height_max = 0.0;
  bool has_height_range = false;
  std::vector<int> atom_types;
  std::vector<int> num_atoms;
  std::vector<double> velocities;
  bool has_file = false;
  std::string add_atom_file;
  double file_velocity = 0.0;
  bool has_file_velocity = false;
  struct FileAtom {
    int type;
    double pos[3];
    double vel[3];
  };
  std::vector<FileAtom> file_atoms;

  void initialize();
  void prepare_subrun(int run_idx);
  bool has_deposition(const std::string& filename);

private:
  static void copy_file(const std::string& in_file, const std::string& out_file);
  void parse_deposition(const char** param, int num_param);
  void analyze_run(const std::string& filename);
  void deposit(const std::string& in_xyz, const std::string& out_xyz);
  void read_file_atoms();
  void initialize_rng();

  std::vector<int> deposited_groups;
  std::mt19937 rng;
  int deposition_count = 0;
  bool has_group = false;
  bool has_vel = false;
};
