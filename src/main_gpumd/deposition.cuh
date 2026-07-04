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
#include <string>
#include <vector>

/*----------------------------------------------------------------------------80
The Deposition class splits a run.in file into sub-run input files according to
the deposition keyword and performs atom insertion between consecutive sub-runs.
------------------------------------------------------------------------------*/

class Deposition {
public:
  std::vector<std::string> subrun_files;

  int interval = 0;
  int atom_type = 0;
  int num_atoms = 1;
  int direction = 2;
  int deposit_runs = 1;
  double velocity = 0.0;
  double height_min = 0.0;
  double height_max = 0.0;
  bool has_height_range = false;

  void initialize();
  void prepare_subrun(int run_idx);
  bool has_deposition(const std::string& filename);

private:
  static std::string trim_comment(const std::string& line);
  static void copy_file(const std::string& in_file, const std::string& out_file);
  void parse_deposition(const char** param, int num_param);
  void split(const std::string& filename);
  void deposit(const std::string& in_xyz, const std::string& out_xyz);

  std::vector<int> deposited_group_label_;
};


