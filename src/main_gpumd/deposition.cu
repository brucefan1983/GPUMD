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
  if (!in.is_open()) {
    std::cout << "Failed to open " << in_file << " for copying." << std::endl;
    exit(1);
  }
  std::ofstream out(out_file, std::ios::binary);
  if (!out.is_open()) {
    std::cout << "Failed to open " << out_file << " for copying." << std::endl;
    exit(1);
  }
  out << in.rdbuf();
  in.close();
  out.close();
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
  if (num_param != 7 && num_param != 8) {
    PRINT_INPUT_ERROR("deposition should have 6 or 7 parameters.\n");
  }

  if (!is_valid_int(param[1], &interval)) {
    PRINT_INPUT_ERROR("deposition interval should be an integer.\n");
  }
  if (interval <= 0) {
    PRINT_INPUT_ERROR("deposition interval should be positive.\n");
  }

  if (!is_valid_int(param[2], &atom_type)) {
    PRINT_INPUT_ERROR("deposition atom_type should be an integer.\n");
  }
  if (atom_type < 0) {
    PRINT_INPUT_ERROR("deposition atom_type should >= 0.\n");
  }

  if (!is_valid_int(param[3], &num_atoms)) {
    PRINT_INPUT_ERROR("deposition num_atoms should be an integer.\n");
  }
  if (num_atoms <= 0) {
    PRINT_INPUT_ERROR("deposition num_atoms should be positive.\n");
  }

  if (!is_valid_int(param[4], &direction)) {
    PRINT_INPUT_ERROR("deposition direction should be an integer.\n");
  }
  if (direction < 0 || direction > 2) {
    PRINT_INPUT_ERROR("deposition direction should be 0 (x), 1 (y), or 2 (z).\n");
  }

  if (!is_valid_real(param[5], &velocity)) {
    PRINT_INPUT_ERROR("deposition velocity should be a real number.\n");
  }

  if (!is_valid_real(param[6], &height_min)) {
    PRINT_INPUT_ERROR("deposition height should be a real number.\n");
  }
  if (height_min <= 0) {
    PRINT_INPUT_ERROR("deposition height_min should be positive.\n");
  }
  if (num_param == 8) {
    if (!is_valid_real(param[7], &height_max)) {
      PRINT_INPUT_ERROR("deposition height_max should be a real number.\n");
    }
    if (height_max < height_min) {
      PRINT_INPUT_ERROR("deposition height_max should >= height_min.\n");
    }
    has_height_range = true;
  } else {
    height_max = height_min;
    has_height_range = false;
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

void Deposition::split(const std::string& filename)
{
  subrun_files.clear();

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

    if (tokens[0] == "deposition") {
      parse_deposition(param.data(), num_param);
      deposition_line = n;
    } else if (tokens[0] == "run") {
      if (num_param != 2) {
        PRINT_INPUT_ERROR("run should have 1 parameter.\n");
      }
      if (!is_valid_int(param[1], &total_steps)) {
        PRINT_INPUT_ERROR("number of steps should be an integer.\n");
      }
      run_line = n;
    }
  }

  if (total_steps % interval != 0) {
    PRINT_INPUT_ERROR("total steps should be divisible by deposition interval.\n");
  }
  deposit_runs = total_steps / interval;
  if (deposit_runs < 1) {
    PRINT_INPUT_ERROR("total steps should be at least the deposition interval.\n");
  }

  for (int i = 0; i < deposit_runs + 1; ++i) {
    std::string filename = "run_" + std::to_string(i) + ".in";
    std::ofstream out(filename);
    if (!out.is_open()) {
      std::cout << "Failed to open " << filename << " for writing." << std::endl;
      exit(1);
    }

    for (size_t n = 0; n < raw_lines.size(); ++n) {
      if (int(n) == deposition_line) {
        continue;
      }

      if (i > 0) {
        std::string line = trim_comment(raw_lines[n]);
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start != std::string::npos) {
          line = line.substr(start);
          std::vector<std::string> tokens = get_tokens(line);
          if (!tokens.empty() && tokens[0] == "velocity") {
            continue;
          }
        }
      }

      if (int(n) == run_line) {
        out << "dump_xyz -1 0 " << interval << " deposition_" << i << ".xyz velocity";
        if (has_model_group_) {
          out << " group";
        }
        out << "\n";
        out << "run " << interval << "\n";
      } else {
        out << raw_lines[n] << "\n";
      }
    }
    out.close();

    subrun_files.emplace_back(filename);
  }

}

void Deposition::deposit(const std::string& input_xyz, const std::string& output_xyz)
{
  std::ifstream input(input_xyz);
  if (!input.is_open()) {
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
        if (group_label > max_group[g]) {
          max_group[g] = group_label;
        }
      }
    }
  }
  input.close();

  if (has_group && deposited_group_label_.empty()) {
    deposited_group_label_.resize(num_group_methods);
    for (int g = 0; g < num_group_methods; ++g) {
      deposited_group_label_[g] = max_group[g] + 1;
    }
  }

  std::string potential_filename = get_filename_potential();
  std::vector<std::string> atom_symbols = get_atom_symbols(potential_filename);
  const std::string symbol = atom_symbols[atom_type];

  std::vector<std::string> lattice_tokens = get_tokens(comment_line);
  for (auto& token : lattice_tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }

  double h[9] = {0.0};
  for (size_t n = 0; n < lattice_tokens.size(); ++n) {
    const std::string lattice_string = "lattice=";
    if (lattice_tokens[n].substr(0, lattice_string.length()) == lattice_string) {
      for (int m = 0; m < 9; ++m) {
        h[m] = get_double_from_token(
          lattice_tokens[n + m].substr(
            (m == 0) ? (lattice_string.length() + 1) : 0,
            (m == 8) ? (lattice_tokens[n + m].length() - 1) : lattice_tokens[n + m].length()),
          __FILE__,
          __LINE__);
      }
      break;
    }
  }

  const double lx = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
  const double ly = std::sqrt(h[3] * h[3] + h[4] * h[4] + h[5] * h[5]);
  const double lz = std::sqrt(h[6] * h[6] + h[7] * h[7] + h[8] * h[8]);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist01(0.0, 1.0);

  std::vector<std::string> new_atom_lines;
  for (int n = 0; n < num_atoms; ++n) {
    double height = height_min;
    if (has_height_range) {
      height = height_min + dist01(gen) * (height_max - height_min);
    }

    double pos_a = 0.0, pos_b = 0.0;
    double x = 0.0, y = 0.0, z = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;

    if (direction == 0) {
      pos_a = dist01(gen) * ly;
      pos_b = dist01(gen) * lz;
      x = height;
      y = pos_a;
      z = pos_b;
      vx = velocity;
    } else if (direction == 1) {
      pos_a = dist01(gen) * lx;
      pos_b = dist01(gen) * lz;
      x = pos_a;
      y = height;
      z = pos_b;
      vy = velocity;
    } else {
      pos_a = dist01(gen) * lx;
      pos_b = dist01(gen) * ly;
      x = pos_a;
      y = pos_b;
      z = height;
      vz = velocity;
    }

    std::ostringstream atom_line;
    atom_line << symbol << " " << x << " " << y << " " << z << " " << vx << " " << vy << " "
              << vz;
    if (has_group) {
      for (int g = 0; g < num_group_methods; ++g) {
        atom_line << " " << deposited_group_label_[g];
      }
    }
    new_atom_lines.emplace_back(atom_line.str());
  }

  std::ofstream out(output_xyz);
  if (!out.is_open()) {
    std::cout << "Failed to open " << output_xyz << " for writing." << std::endl;
    exit(1);
  }

  out << original_num_atoms + num_atoms << "\n";
  out << comment_line << "\n";
  for (const auto& atom_line : atom_lines) {
    out << atom_line << "\n";
  }
  for (const auto& new_line : new_atom_lines) {
    out << new_line << "\n";
  }
  out.close();
}

void Deposition::initialize()
{
  copy_file("run.in", "run.in.original");
  copy_file("model.xyz", "model.xyz.original");

  std::ifstream model_input("model.xyz");
  if (model_input.is_open()) {
    std::string line;
    std::getline(model_input, line); // number of atoms
    std::getline(model_input, line); // comment line with properties
    int num_group_methods = 0;
    has_model_group_ = has_group_property(line, num_group_methods);
    model_input.close();
  }

  split("run.in.original");
  printf("Split run.in into %zu sub-run input files.\n", subrun_files.size());
}

void Deposition::prepare_subrun(int run_idx)
{
  printf(
    "Running sub-run %d / %zu (%s).\n", run_idx + 1, subrun_files.size(),
    subrun_files[run_idx].c_str());
  fflush(stdout);

  copy_file(subrun_files[run_idx], "run.in");

  if (run_idx > 0) {
    std::string previous_xyz = "deposition_" + std::to_string(run_idx - 1) + ".xyz";
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
    if (!tokens_without_comments.empty() && tokens_without_comments[0] == "deposition") {
      input.close();
      return true;
    }
  }

  input.close();
  return false;
}
