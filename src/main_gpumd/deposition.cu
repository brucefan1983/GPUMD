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
#include "utilities/compact_nep.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

const std::map<std::string, double> MASS_TABLE{
  {"H", 1.0080000000},
  {"He", 4.0026020000},
  {"Li", 6.9400000000},
  {"Be", 9.0121831000},
  {"B", 10.8100000000},
  {"C", 12.0110000000},
  {"N", 14.0070000000},
  {"O", 15.9990000000},
  {"F", 18.9984031630},
  {"Ne", 20.1797000000},
  {"Na", 22.9897692800},
  {"Mg", 24.3050000000},
  {"Al", 26.9815385000},
  {"Si", 28.0850000000},
  {"P", 30.9737619980},
  {"S", 32.0600000000},
  {"Cl", 35.4500000000},
  {"Ar", 39.9480000000},
  {"K", 39.0983000000},
  {"Ca", 40.0780000000},
  {"Sc", 44.9559080000},
  {"Ti", 47.8670000000},
  {"V", 50.9415000000},
  {"Cr", 51.9961000000},
  {"Mn", 54.9380440000},
  {"Fe", 55.8450000000},
  {"Co", 58.9331940000},
  {"Ni", 58.6934000000},
  {"Cu", 63.5460000000},
  {"Zn", 65.3800000000},
  {"Ga", 69.7230000000},
  {"Ge", 72.6300000000},
  {"As", 74.9215950000},
  {"Se", 78.9710000000},
  {"Br", 79.9040000000},
  {"Kr", 83.7980000000},
  {"Rb", 85.4678000000},
  {"Sr", 87.6200000000},
  {"Y", 88.9058400000},
  {"Zr", 91.2240000000},
  {"Nb", 92.9063700000},
  {"Mo", 95.9500000000},
  {"Tc", 98},
  {"Ru", 101.0700000000},
  {"Rh", 102.9055000000},
  {"Pd", 106.4200000000},
  {"Ag", 107.8682000000},
  {"Cd", 112.4140000000},
  {"In", 114.8180000000},
  {"Sn", 118.7100000000},
  {"Sb", 121.7600000000},
  {"Te", 127.6000000000},
  {"I", 126.9044700000},
  {"Xe", 131.2930000000},
  {"Cs", 132.9054519600},
  {"Ba", 137.3270000000},
  {"La", 138.9054700000},
  {"Ce", 140.1160000000},
  {"Pr", 140.9076600000},
  {"Nd", 144.2420000000},
  {"Pm", 145},
  {"Sm", 150.3600000000},
  {"Eu", 151.9640000000},
  {"Gd", 157.2500000000},
  {"Tb", 158.9253500000},
  {"Dy", 162.5000000000},
  {"Ho", 164.9303300000},
  {"Er", 167.2590000000},
  {"Tm", 168.9342200000},
  {"Yb", 173.0450000000},
  {"Lu", 174.9668000000},
  {"Hf", 178.4900000000},
  {"Ta", 180.9478800000},
  {"W", 183.8400000000},
  {"Re", 186.2070000000},
  {"Os", 190.2300000000},
  {"Ir", 192.2170000000},
  {"Pt", 195.0840000000},
  {"Au", 196.9665690000},
  {"Hg", 200.5920000000},
  {"Tl", 204.3800000000},
  {"Pb", 207.2000000000},
  {"Bi", 208.9804000000},
  {"Po", 210},
  {"At", 210},
  {"Rn", 222},
  {"Fr", 223},
  {"Ra", 226},
  {"Ac", 227},
  {"Th", 232.0377000000},
  {"Pa", 231.0358800000},
  {"U", 238.0289100000},
  {"Np", 237},
  {"Pu", 244},
  {"Am", 243},
  {"Cm", 247},
  {"Bk", 247},
  {"Cf", 251},
  {"Es", 252},
  {"Fm", 257},
  {"Md", 258},
  {"No", 259},
  {"Lr", 262}};

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

static int get_deposit_atom_type(
  const std::string& token, const std::vector<std::string>& atom_symbols)
{
  int atom_type = 0;
  if (is_valid_int(token.c_str(), &atom_type)) {
    PRINT_INPUT_ERROR("deposit atom species should be specified using an element symbol.\n");
  }

  for (int n = 0; n < static_cast<int>(atom_symbols.size()); ++n) {
    if (atom_symbols[n] == token) {
      return n;
    }
  }

  std::string error = "deposit atom species " + token + " is not supported by the potential file.";
  PRINT_INPUT_ERROR(error.c_str());
  return -1;
}

void Deposition::parse_deposition(const char** param, int num_param)
{
  for (int i = 0; i < num_param; ++i) {
    if (std::string(param[i])[0] == '#') {
      num_param = i;
      break;
    }
  }

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
    masses.clear();
    while (idx < num_param) {
      if (idx + 3 > num_param) {
        PRINT_INPUT_ERROR("deposit atom species requires element, number, and velocity.\n");
      }

      int atom_type = get_deposit_atom_type(param[idx], atom_symbols);
      register_compact_nep_required_species(atom_symbols[atom_type]);
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

      double mass = 0.0;
      if (idx < num_param && is_valid_real(param[idx], &mass)) {
        if (mass <= 0) {
          PRINT_INPUT_ERROR("deposit mass should be positive.\n");
        }
        ++idx;
      }

      atom_types.emplace_back(atom_type);
      num_atoms.emplace_back(number);
      velocities.emplace_back(velocity);
      masses.emplace_back(mass);
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

    if (direction < 0) {
      if (idx != num_param) {
        PRINT_INPUT_ERROR("deposit file mode with direction=-1 should not have parameters after the file name.\n");
      }
      has_file_velocity = false;
    } else {
      if (idx >= num_param) {
        PRINT_INPUT_ERROR("deposit file mode with a direction should be followed by a velocity.\n");
      }
      if (!is_valid_real(param[idx], &file_velocity)) {
        PRINT_INPUT_ERROR("deposit file velocity should be a real number.\n");
      }
      has_file_velocity = true;
      ++idx;
      if (idx != num_param) {
        PRINT_INPUT_ERROR("deposit file mode should not have extra parameters after the velocity.\n");
      }
    }
    read_file_atoms();
  }
}

static std::vector<std::pair<std::string, int>> get_property_list(const std::string& comment_line)
{
  std::vector<std::pair<std::string, int>> properties;

  const std::string properties_marker = "Properties=";
  size_t value_start = comment_line.find(properties_marker);
  if (value_start == std::string::npos) {
    return properties;
  }
  value_start += properties_marker.length();

  const size_t value_end = comment_line.find_first_of(" \t", value_start);
  std::string value = comment_line.substr(
    value_start, value_end == std::string::npos ? value_end : value_end - value_start);

  for (auto& letter : value) {
    letter = (letter == ':') ? ' ' : std::tolower(letter);
  }

  const std::vector<std::string> tokens = get_tokens(value);
  for (size_t k = 0; k + 2 < tokens.size(); k += 3) {
    properties.emplace_back(tokens[k], get_int_from_token(tokens[k + 2], __FILE__, __LINE__));
  }
  return properties;
}

static int find_property(
  const std::vector<std::pair<std::string, int>>& properties, const std::string& name)
{
  for (const auto& property : properties) {
    if (property.first == name) {
      return property.second;
    }
  }
  return 0;
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
    std::vector<std::string> tokens = get_tokens(raw_lines[n]);
    if (tokens.empty()) {
      continue;
    }

    std::vector<const char*> param(tokens.size());
    for (int k = 0; k < tokens.size(); ++k) {
      param[k] = tokens[k].c_str();
    }
    const int num_param = tokens.size();

    if (tokens[0] == "deposit") {
      if (deposition_line >= 0) {
        PRINT_INPUT_ERROR("deposit should appear only once in run.in.\n");
      }
      parse_deposition(param.data(), num_param);
      deposition_line = n;
    } else if (tokens[0] == "run") {
      if (run_line >= 0) {
        PRINT_INPUT_ERROR("run should appear only once in run.in when deposit is used.\n");
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
          "dump_xyz " + std::to_string(interval) + " deposited_" + std::to_string(dump_index) +
          ".xyz" + (has_mass ? " mass" : "") + " velocity" + (has_group ? " group_labels" : ""));
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
    const size_t comment_pos = raw_line.find('#');
    const std::string line = (comment_pos == std::string::npos) ? raw_line : raw_line.substr(0, comment_pos);
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.empty()) {
      continue;
    }

    if (tokens.size() != 7 && tokens.size() != 8) {
      PRINT_INPUT_ERROR(
        "deposit file should have 7 columns: element x y z vx vy vz (plus an optional mass column).\n");
    }

    FileAtom fa;
    fa.type = get_deposit_atom_type(tokens[0], atom_symbols);
    register_compact_nep_required_species(atom_symbols[fa.type]);
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
    fa.mass = 0.0;
    if (tokens.size() == 8) {
      if (!is_valid_real(tokens[7].c_str(), &fa.mass)) {
        PRINT_INPUT_ERROR("deposit file mass should be a real number.\n");
      }
      if (fa.mass <= 0) {
        PRINT_INPUT_ERROR("deposit file mass should be positive.\n");
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

  const std::vector<std::pair<std::string, int>> properties = get_property_list(comment_line);
  if (properties.empty()) {
    PRINT_INPUT_ERROR("deposit requires the Properties field in the second line of the xyz file.\n");
  }
  const int num_group_methods = find_property(properties, "group");
  const bool has_group = num_group_methods > 0;
  int group_offset = -1;
  int column_offset = 0;
  for (const auto& property : properties) {
    if (property.first == "group") {
      group_offset = column_offset;
      break;
    }
    column_offset += property.second;
  }

  int total_new_atoms = 0;
  if (has_file) {
    total_new_atoms = file_atoms.size();
  } else {
    for (const int n : num_atoms) {
      total_new_atoms += n;
    }
  }

  double h[9] = {0.0};
  const std::string lattice_marker = "Lattice=\"";
  const size_t lattice_pos = comment_line.find(lattice_marker);
  if (lattice_pos != std::string::npos) {
    const size_t value_start = lattice_pos + lattice_marker.length();
    const size_t value_end = comment_line.find("\"", value_start);
    const std::string lattice_values = comment_line.substr(value_start, value_end - value_start);
    const std::vector<std::string> lattice_tokens = get_tokens(lattice_values);
    if (lattice_tokens.size() < 9) {
      PRINT_INPUT_ERROR("Lattice in the xyz file should have 9 numbers.\n");
    }
    for (int m = 0; m < 9; ++m) {
      h[m] = get_double_from_token(lattice_tokens[m], __FILE__, __LINE__);
    }
  }

  double l[3];
  for (int m = 0; m < 3; ++m) {
    const double* v = h + 3 * m;
    l[m] = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

#ifdef DEBUG
  rng.seed(12345678 + deposition_count);
#endif
  ++deposition_count;

  std::uniform_real_distribution<double> dist01(0.0, 1.0);

  std::ofstream out(output_xyz);
  if (!out) {
    std::cout << "Failed to open " << output_xyz << " for writing." << std::endl;
    exit(1);
  }

  out << original_num_atoms + total_new_atoms << "\n";
  out << comment_line << "\n";

  std::vector<int> max_group(num_group_methods, -1);
  for (int n = 0; n < original_num_atoms; ++n) {
    std::getline(input, line);
    out << line << "\n";
    if (has_group) {
      std::vector<std::string> tokens = get_tokens(line);
      for (int g = 0; g < num_group_methods; ++g) {
        const int group_label = get_int_from_token(tokens[group_offset + g], __FILE__, __LINE__);
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

  auto sample_height = [&]() {
    return has_height_range ? height_min + dist01(rng) * (height_max - height_min) : height_min;
  };

  auto add_atom_line = [&](int atom_type, const double pos[3], const double vel[3],
                           const double mass) {
    if (atom_type >= int(atom_symbols.size())) {
      PRINT_INPUT_ERROR("deposit atom_type exceeds the number of atom types in the potential file.\n");
    }
    const std::string symbol = atom_symbols[atom_type];

    std::ostringstream atom_line;
    atom_line << std::setprecision(10);

    for (size_t p = 0; p < properties.size(); ++p) {
      const std::string& name = properties[p].first;
      const int length = properties[p].second;
      const char* sep = (p == 0) ? "" : " ";
      if (name == "species") {
        atom_line << sep << symbol;
      } else if (name == "pos") {
        for (int d = 0; d < 3; ++d) {
          atom_line << ((p == 0 && d == 0) ? "" : " ") << pos[d];
        }
      } else if (name == "mass") {
        atom_line << sep << ((mass > 0.0) ? mass : MASS_TABLE.at(symbol));
      } else if (name == "vel") {
        for (int d = 0; d < 3; ++d) {
          atom_line << ((p == 0 && d == 0) ? "" : " ") << vel[d];
        }
      } else if (name == "group") {
        for (int g = 0; g < length; ++g) {
          atom_line << ((p == 0 && g == 0) ? "" : " ") << deposited_groups[g];
        }
      } else {
        const std::string error =
          "deposit does not support the '" + name + "' property in the xyz file.";
        PRINT_INPUT_ERROR(error.c_str());
      }
    }
    out << atom_line.str() << "\n";
  };

  if (has_file) {
    for (const auto& fa : file_atoms) {
      double pos[3] = {fa.pos[0], fa.pos[1], fa.pos[2]};
      if (direction >= 0) {
        pos[direction] = sample_height();
      }

      double vel[3] = {fa.vel[0], fa.vel[1], fa.vel[2]};
      if (has_file_velocity) {
        vel[direction] = file_velocity;
      }

      add_atom_line(fa.type, pos, vel, fa.mass);
    }
  } else {
    for (size_t s = 0; s < atom_types.size(); ++s) {
      const double velocity = velocities[s];
      for (int n = 0; n < num_atoms[s]; ++n) {
        double pos[3] = {dist01(rng) * l[0], dist01(rng) * l[1], dist01(rng) * l[2]};
        pos[direction] = sample_height();

        double vel[3] = {0.0, 0.0, 0.0};
        vel[direction] = velocity;

        add_atom_line(atom_types[s], pos, vel, masses[s]);
      }
    }
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

  std::string potential_filename = get_filename_potential();
  atom_symbols = get_atom_symbols(potential_filename);

  std::ifstream model_input("model.xyz");
  if (model_input.is_open()) {
    std::string line;
    std::getline(model_input, line);
    std::getline(model_input, line);
    const std::vector<std::pair<std::string, int>> properties = get_property_list(line);
    has_group = find_property(properties, "group") > 0;
    has_vel = find_property(properties, "vel") > 0;
    has_mass = find_property(properties, "mass") > 0;
    model_input.close();
  }

  analyze_run("run.in.original");
  printf("Split run.in into %d sub-runs.\n", num_subruns);
}

void Deposition::prepare_subrun(int run_idx)
{
  printf(
    "Running sub-run %d / %d.\n", run_idx + 1, num_subruns);
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
    if (!tokens.empty() && tokens[0] == "deposit") {
      input.close();
      return true;
    }
  }

  input.close();
  return false;
}
