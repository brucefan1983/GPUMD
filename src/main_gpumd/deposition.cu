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
Split a run.in file into sub-run input files according to the deposition
keyword and perform deposition between consecutive sub-runs.
------------------------------------------------------------------------------*/

#include "deposition.cuh"
#include "model/read_xyz.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

void Deposition::copy_file(const std::string& in_file, const std::string& out_file)
{
  std::ifstream in(in_file, std::ios::binary);
  std::ofstream out(out_file, std::ios::binary);
  if (!in || !out) {
    std::cout << "Failed to open " << in_file << " for copy to " << out_file << "." << std::endl;
    exit(1);
  }
  out << in.rdbuf();
}

std::string Deposition::trim_comment(const std::string& line)
{
  size_t pos = line.find('#');
  if (pos == std::string::npos) {
    return line;
  }
  return line.substr(0, pos);
}

void Deposition::parse_deposition(const char** param, int num_param)
{
  if (num_param < 6) {
    PRINT_INPUT_ERROR("deposit should have at least 5 parameters.\n");
  }

  if (!is_valid_int(param[1], &interval)) {
    PRINT_INPUT_ERROR("deposit interval should be an integer.\n");
  }
  if (interval <= 0) {
    PRINT_INPUT_ERROR("deposit interval should be positive.\n");
  }

  if (!is_valid_int(param[2], &direction)) {
    PRINT_INPUT_ERROR("deposit direction should be -1 or 0 (x), 1 (y), 2 (z).\n");
  }
  if (direction < -1 || direction > 2) {
    PRINT_INPUT_ERROR("deposit direction should be -1 or 0 (x), 1 (y), 2 (z).\n");
  }

  if (!is_valid_real(param[3], &height_min)) {
    PRINT_INPUT_ERROR("deposit height should be a real number.\n");
  }
  if (direction >= 0 && height_min <= 0) {
    PRINT_INPUT_ERROR("deposit height_min should be positive.\n");
  }

  int idx = 4;
  if (std::string(param[idx]) == "atom" || std::string(param[idx]) == "file") {
    height_max = height_min;
    has_height_range = false;
  } else {
    if (!is_valid_real(param[idx], &height_max)) {
      PRINT_INPUT_ERROR("deposit height_max should be a real number, 'atom', or 'file'.\n");
    }
    if (direction >= 0 && height_max < height_min) {
      PRINT_INPUT_ERROR("deposit height_max should >= height_min.\n");
    }
    has_height_range = true;
    ++idx;
  }

  if (idx >= num_param || (std::string(param[idx]) != "atom" && std::string(param[idx]) != "file")) {
    PRINT_INPUT_ERROR("deposit should have 'atom' or 'file' keyword.\n");
  }

  if (std::string(param[idx]) == "atom") {
    if (direction == -1) {
      PRINT_INPUT_ERROR("deposit with direction=-1 does not support 'atom' mode.\n");
    }
    has_file = false;
    ++idx;

    atom_types.clear();
    num_atoms.clear();
    velocities.clear();
    while (idx < num_param) {
      if (idx + 2 > num_param) {
        PRINT_INPUT_ERROR("deposit atom species requires type, number, and velocity.\n");
      }

      int atom_type = 0;
      if (!is_valid_int(param[idx], &atom_type)) {
        PRINT_INPUT_ERROR("deposit atom_type should be an integer.\n");
      }
      if (atom_type < 0) {
        PRINT_INPUT_ERROR("deposit atom_type should >= 0.\n");
      }
      ++idx;

      int number = 0;
      if (!is_valid_int(param[idx], &number)) {
        PRINT_INPUT_ERROR("deposit num_atoms should be an integer.\n");
      }
      if (number <= 0) {
        PRINT_INPUT_ERROR("deposit num_atoms should be positive.\n");
      }
      ++idx;

      double velocity = 0.0;
      if (!is_valid_real(param[idx], &velocity)) {
        PRINT_INPUT_ERROR("deposit velocity should be a real number.\n");
      }
      ++idx;

      atom_types.emplace_back(atom_type);
      num_atoms.emplace_back(number);
      velocities.emplace_back(velocity);
    }

    if (atom_types.empty()) {
      PRINT_INPUT_ERROR("deposit should have at least one atom species.\n");
    }
  } else {
    has_file = true;
    ++idx;
    if (idx >= num_param) {
      PRINT_INPUT_ERROR("deposit file keyword should be followed by a file name.\n");
    }
    add_atom_file = param[idx];
    ++idx;
    if (idx != num_param) {
      PRINT_INPUT_ERROR("deposit file mode should not have extra parameters after the file name.\n");
    }
    read_file_atoms();
  }
}

static bool has_group_property(const std::string& comment_line, int& num_group_methods)
{
  const std::string group_marker = "group:I:";
  const size_t pos = comment_line.find(group_marker);
  if (pos == std::string::npos) {
    return false;
  }
  num_group_methods = std::stoi(comment_line.substr(pos + group_marker.length()));
  return num_group_methods > 0;
}

static bool has_velocity_property(const std::string& comment_line)
{
  const std::string velocity_marker = "vel:R:3";
  return comment_line.find(velocity_marker) != std::string::npos;
}

void Deposition::analyze_run(const std::string& filename)
{
  subrun_lines.clear();

  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << "." << std::endl;
    exit(1);
  }
  std::vector<std::string> raw_lines;
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    raw_lines.emplace_back(raw_line);
  }
  input.close();

  int deposition_line = -1;
  int run_line = -1;
  int total_steps = 0;

  for (int n = 0; n < raw_lines.size(); ++n) {
    std::string line = trim_comment(raw_lines[n]);
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    line = line.substr(start);

    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.empty()) {
      continue;
    }

    std::vector<const char*> param(tokens.size());
    for (int k = 0; k < tokens.size(); ++k) {
      param[k] = tokens[k].c_str();
    }
    const int num_param = tokens.size();

    if (tokens[0] == "deposit") {
      parse_deposition(param.data(), num_param);
      deposition_line = n;
    } else if (tokens[0] == "run") {
      if (!is_valid_int(param[1], &total_steps)) {
        PRINT_INPUT_ERROR("number of steps should be an integer.\n");
      }
      run_line = n;
    }
  }

  if (total_steps % interval != 0) {
    PRINT_INPUT_ERROR("total steps should be divisible by deposition interval.\n");
  }
  const int deposit_runs = total_steps / interval;
  if (deposit_runs < 1) {
    PRINT_INPUT_ERROR("total steps should be at least the deposition interval.\n");
  }

  num_subruns = has_vel ? deposit_runs : deposit_runs + 1;
  for (int i = 0; i < num_subruns; ++i) {
    std::vector<std::string> lines;
    const int dump_index = has_vel ? (i + 1) : i;

    for (size_t n = 0; n < raw_lines.size(); ++n) {
      if (int(n) == deposition_line) {
        continue;
      }

      if (int(n) == run_line) {
        lines.emplace_back(
          "dump_xyz -1 0 " + std::to_string(interval) + " deposited_" +
          std::to_string(dump_index) + ".xyz velocity" + (has_group ? " group" : ""));
        lines.emplace_back("run " + std::to_string(interval));
      } else {
        lines.emplace_back(raw_lines[n]);
      }
    }
    subrun_lines.emplace_back(std::move(lines));
  }
}

void Deposition::read_file_atoms()
{
  std::ifstream input(add_atom_file);
  if (!input.is_open()) {
    std::cout << "Failed to open " << add_atom_file << "." << std::endl;
    exit(1);
  }

  file_atoms.clear();
  std::string raw_line;
  while (std::getline(input, raw_line)) {
    std::string line = trim_comment(raw_line);
    size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    line = line.substr(start);

    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.empty()) {
      continue;
    }

    if (tokens.size() != 7) {
      PRINT_INPUT_ERROR("deposit file should have 7 columns: type x y z vx vy vz.\n");
    }

    FileAtom fa;
    if (!is_valid_int(tokens[0].c_str(), &fa.type)) {
      PRINT_INPUT_ERROR("deposit file atom_type should be an integer.\n");
    }
    if (fa.type < 0) {
      PRINT_INPUT_ERROR("deposit file atom_type should >= 0.\n");
    }
    for (int d = 0; d < 3; ++d) {
      if (!is_valid_real(tokens[1 + d].c_str(), &fa.pos[d])) {
        PRINT_INPUT_ERROR("deposit file position should be a real number.\n");
      }
    }
    for (int d = 0; d < 3; ++d) {
      if (!is_valid_real(tokens[4 + d].c_str(), &fa.vel[d])) {
        PRINT_INPUT_ERROR("deposit file velocity should be a real number.\n");
      }
    }

    file_atoms.emplace_back(fa);
  }
  input.close();

  if (file_atoms.empty()) {
    PRINT_INPUT_ERROR("deposit file should have at least one atom.\n");
  }
}

void Deposition::deposit(const std::string& input_xyz, const std::string& output_xyz)
{
  std::ifstream input(input_xyz);
  if (!input) {
    std::cout << "Failed to open " << input_xyz << "." << std::endl;
    exit(1);
  }

  std::string line;
  std::getline(input, line);
  int original_num_atoms = get_int_from_token(get_tokens(line)[0], __FILE__, __LINE__);

  std::getline(input, line);
  std::string comment_line = line;

  int num_group_methods = 0;
  const bool has_group = has_group_property(comment_line, num_group_methods);

  std::vector<std::string> atom_lines;
  std::vector<int> max_group(num_group_methods, -1);
  for (int n = 0; n < original_num_atoms; ++n) {
    std::getline(input, line);
    atom_lines.emplace_back(line);
    if (has_group) {
      std::vector<std::string> tokens = get_tokens(line);
      for (int g = 0; g < num_group_methods; ++g) {
        const int col = int(tokens.size()) - num_group_methods + g;
        const int group_label = get_int_from_token(tokens[col], __FILE__, __LINE__);
        max_group[g] = std::max(max_group[g], group_label);
      }
    }
  }
  input.close();

  if (has_group && deposited_groups.empty()) {
    deposited_groups.resize(num_group_methods);
    for (int g = 0; g < num_group_methods; ++g) {
      deposited_groups[g] = max_group[g] + 1;
    }
  }

  std::string potential_filename = get_filename_potential();
  std::vector<std::string> atom_symbols = get_atom_symbols(potential_filename);

  double h[9] = {0.0};
  const std::string lattice_marker = "Lattice=\"";
  const size_t lattice_pos = comment_line.find(lattice_marker);
  if (lattice_pos != std::string::npos) {
    const size_t value_start = lattice_pos + lattice_marker.length();
    const size_t value_end = comment_line.find("\"", value_start);
    const std::string lattice_values = comment_line.substr(value_start, value_end - value_start);
    const std::vector<std::string> lattice_tokens = get_tokens(lattice_values);
    for (int m = 0; m < 9; ++m) {
      h[m] = get_double_from_token(lattice_tokens[m], __FILE__, __LINE__);
    }
  }

  double l[3];
  for (int m = 0; m < 3; ++m) {
    l[m] = std::sqrt(h[3 * m] * h[3 * m] + h[3 * m + 1] * h[3 * m + 1] + h[3 * m + 2] * h[3 * m + 2]);
  }

#ifdef DEBUG
  rng.seed(12345678 + deposition_count);
#endif
  ++deposition_count;

  std::uniform_real_distribution<double> dist01(0.0, 1.0);

  std::vector<std::string> new_atom_lines;

  auto add_atom_line = [&](int atom_type, const double pos[3], const double vel[3]) {
    const std::string symbol = atom_symbols[atom_type];

    std::ostringstream atom_line;
    atom_line << symbol << " " << pos[0] << " " << pos[1] << " " << pos[2] << " " << vel[0]
              << " " << vel[1] << " " << vel[2];
    if (has_group) {
      for (int g = 0; g < num_group_methods; ++g) {
        atom_line << " " << deposited_groups[g];
      }
    }
    new_atom_lines.emplace_back(atom_line.str());
  };

  if (has_file) {
    for (const auto& fa : file_atoms) {
      double pos[3];
      for (int d = 0; d < 3; ++d) {
        pos[d] = fa.pos[d];
      }
      if (direction >= 0) {
        double height = height_min;
        if (has_height_range) {
          height = height_min + dist01(rng) * (height_max - height_min);
        }
        pos[direction] = height;
      }

      add_atom_line(fa.type, pos, fa.vel);
    }
  } else {
    for (size_t s = 0; s < atom_types.size(); ++s) {
      const double velocity = velocities[s];
      for (int n = 0; n < num_atoms[s]; ++n) {
        double height = height_min;
        if (has_height_range) {
          height = height_min + dist01(rng) * (height_max - height_min);
        }

        double pos[3] = {dist01(rng) * l[0], dist01(rng) * l[1], dist01(rng) * l[2]};
        pos[direction] = height;

        double vel[3] = {0.0, 0.0, 0.0};
        vel[direction] = velocity;

        add_atom_line(atom_types[s], pos, vel);
      }
    }
  }

  std::ofstream out(output_xyz);
  if (!out) {
    std::cout << "Failed to open " << output_xyz << " for writing." << std::endl;
    exit(1);
  }

  int total_new_atoms = 0;
  if (has_file) {
    total_new_atoms = file_atoms.size();
  } else {
    for (const int n : num_atoms) {
      total_new_atoms += n;
    }
  }
  out << original_num_atoms + total_new_atoms << "\n";
  out << comment_line << "\n";
  for (const auto& atom_line : atom_lines) {
    out << atom_line << "\n";
  }
  for (const auto& new_line : new_atom_lines) {
    out << new_line << "\n";
  }
}

void Deposition::initialize_rng()
{
#ifndef DEBUG
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
}

void Deposition::initialize()
{
  copy_file("run.in", "run.in.original");
  copy_file("model.xyz", "model.xyz.original");

  initialize_rng();

  std::ifstream model_input("model.xyz");
  if (model_input.is_open()) {
    std::string line;
    std::getline(model_input, line);
    std::getline(model_input, line);
    int num_group_methods = 0;
    has_group = has_group_property(line, num_group_methods);
    has_vel = has_velocity_property(line);
    model_input.close();
  }

  analyze_run("run.in.original");
  printf("Split run.in into %zu sub-runs.\n", num_subruns);
}

void Deposition::prepare_subrun(int run_idx)
{
  printf(
    "Running sub-run %d / %zu.\n", run_idx + 1, num_subruns);
  fflush(stdout);

  std::ofstream out("run.in");
  if (!out.is_open()) {
    std::cout << "Failed to open run.in for writing." << std::endl;
    exit(1);
  }
  for (const auto& line : subrun_lines[run_idx]) {
    out << line << "\n";
  }
  out.close();

  if (has_vel && run_idx == 0) {
    deposit("model.xyz.original", "model.xyz");
  }

  if (run_idx > 0) {
    const int previous_index = has_vel ? run_idx: (run_idx - 1);
    std::string previous_xyz = "deposited_" + std::to_string(previous_index) + ".xyz";
    deposit(previous_xyz, "model.xyz");
  }
}

bool Deposition::has_deposition(const std::string& filename)
{
  std::ifstream input(filename);
  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << "." << std::endl;
    exit(1);
  }

  while (input.peek() != EOF) {
    std::vector<std::string> tokens = get_tokens(input);
    std::vector<std::string> tokens_without_comments;
    for (const auto& t : tokens) {
      if (t[0] != '#') {
        tokens_without_comments.emplace_back(t);
      } else {
        break;
      }
    }
    if (!tokens_without_comments.empty() && tokens_without_comments[0] == "deposit") {
      input.close();
      return true;
    }
  }

  input.close();
  return false;
}
