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
The class defining the simulation model.
------------------------------------------------------------------------------*/

#include "atom.cuh"
#include "box.cuh"
#include "group.cuh"
#include "read_xyz.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include <vector>
#include <cmath>
#include <cstdint>

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


namespace {

inline uint64_t morton_encode(
  double x, double y, double z,
  double min_x, double max_x,
  double min_y, double max_y,
  double min_z, double max_z,
  int bits) {
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    double range_z = max_z - min_z;
    uint64_t ix = 0, iy = 0, iz = 0;
    if (range_x > 1e-12) {
        double t = (x - min_x) / range_x;
        t = std::max(0.0, std::min(1.0, t));
        ix = static_cast<uint64_t>(t * ((1ULL << bits) - 1));
    }
    if (range_y > 1e-12) {
        double t = (y - min_y) / range_y;
        t = std::max(0.0, std::min(1.0, t));
        iy = static_cast<uint64_t>(t * ((1ULL << bits) - 1));
    }
    if (range_z > 1e-12) {
        double t = (z - min_z) / range_z;
        t = std::max(0.0, std::min(1.0, t));
        iz = static_cast<uint64_t>(t * ((1ULL << bits) - 1));
    }

    uint64_t code = 0;
    for (int b = 0; b < bits; ++b) {
        uint64_t bit_x = (ix >> b) & 1ULL;
        uint64_t bit_y = (iy >> b) & 1ULL;
        uint64_t bit_z = (iz >> b) & 1ULL;
        code |= (bit_x << (3 * b)) | (bit_y << (3 * b + 1)) | (bit_z << (3 * b + 2));
    }
    return code;
}

void sort_position(
  const int num_atoms,
  const double* x, 
  const double* y, 
  const double* z,
  const double* vx,
  const double* vy,
  const double* vz,
  const std::string* atom_symbol,
  const int* type,
  const double* mass,
  const float* charge,
  const std::vector<Group>& group,
  double* x_sorted, 
  double* y_sorted, 
  double* z_sorted,
  double* vx_sorted,
  double* vy_sorted,
  double* vz_sorted,
  std::string* atom_symbol_sorted,
  int* type_sorted,
  double* mass_sorted,
  float* charge_sorted,
  std::vector<Group>& group_sorted) {       
  if (num_atoms <= 0) return;

  double min_x = x[0], max_x = x[0];
  double min_y = y[0], max_y = y[0];
  double min_z = z[0], max_z = z[0];
  for (int i = 1; i < num_atoms; ++i) {
    if (x[i] < min_x) min_x = x[i];
    if (x[i] > max_x) max_x = x[i];
    if (y[i] < min_y) min_y = y[i];
    if (y[i] > max_y) max_y = y[i];
    if (z[i] < min_z) min_z = z[i];
    if (z[i] > max_z) max_z = z[i];
  }

  const double eps = 1e-12;
  if (max_x - min_x < eps) { max_x = min_x + 1.0; }
  if (max_y - min_y < eps) { max_y = min_y + 1.0; }
  if (max_z - min_z < eps) { max_z = min_z + 1.0; }

  const int bits = 20;

  std::vector<std::pair<uint64_t, int>> pairs(num_atoms);
  for (int i = 0; i < num_atoms; ++i) {
    uint64_t code = morton_encode(x[i], y[i], z[i], min_x, max_x, min_y, max_y, min_z, max_z, bits);
    pairs[i] = {code, i};
  }

  std::sort(pairs.begin(), pairs.end(), 
    [](const std::pair<uint64_t, int>& a, const std::pair<uint64_t, int>& b) { return a.first < b.first; });

  for (int i = 0; i < num_atoms; ++i) {
    int orig_idx = pairs[i].second;
    x_sorted[i] = x[orig_idx];
    y_sorted[i] = y[orig_idx];
    z_sorted[i] = z[orig_idx];
    vx_sorted[i] = vx[orig_idx];
    vy_sorted[i] = vy[orig_idx];
    vz_sorted[i] = vz[orig_idx];
    atom_symbol_sorted[i] = atom_symbol[orig_idx];
    type_sorted[i] = type[orig_idx];
    mass_sorted[i] = mass[orig_idx];
    charge_sorted[i] = charge[orig_idx];
    for (int m = 0; m < group.size(); ++m) {
      group_sorted[m].cpu_label[i] = group[m].cpu_label[orig_idx];
    }
  }
}

}

static void read_xyz_line_1(std::ifstream& input, int& N)
{
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 1) {
    PRINT_INPUT_ERROR("The first line for the xyz file should have one value.");
  }
  N = get_int_from_token(tokens[0], __FILE__, __LINE__);
  if (N < 2) {
    PRINT_INPUT_ERROR("Number of atoms should >= 2.");
  } else {
    printf("Number of atoms is %d.\n", N);
  }
}

static void read_xyz_line_2(
  std::ifstream& input,
  Box& box,
  int& has_velocity_in_xyz,
  bool& has_mass,
  bool& has_charge,
  int& num_columns,
  int* property_offset,
  std::vector<Group>& group)
{
  std::vector<std::string> tokens = get_tokens_without_unwanted_spaces(input);
  for (auto& token : tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }

  box.pbc_x = box.pbc_y = box.pbc_z = 1; // default is periodic
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string tmp_string = "pbc=";
    if (tokens[n].substr(0, tmp_string.length()) == tmp_string) {
      if (tokens[n].back() == 't') {
        box.pbc_x = 1;
      } else if (tokens[n].back() == 'f') {
        box.pbc_x = 0;
      } else {
        PRINT_INPUT_ERROR("periodic boundary in x direction should be T or F.");
      }
      if (tokens[n + 1] == "t") {
        box.pbc_y = 1;
      } else if (tokens[n + 1] == "f") {
        box.pbc_y = 0;
      } else {
        PRINT_INPUT_ERROR("periodic boundary in y direction should be T or F.");
      }
      if (tokens[n + 2].front() == 't') {
        box.pbc_z = 1;
      } else if (tokens[n + 2].front() == 'f') {
        box.pbc_z = 0;
      } else {
        PRINT_INPUT_ERROR("periodic boundary in z direction should be T or F.");
      }
    }
  }
  printf("Use %s boundary conditions along x.\n", (box.pbc_x == 1) ? "periodic" : "free");
  printf("Use %s boundary conditions along y.\n", (box.pbc_y == 1) ? "periodic" : "free");
  printf("Use %s boundary conditions along z.\n", (box.pbc_z == 1) ? "periodic" : "free");

  // box matrix
  bool has_lattice_in_exyz = false;
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string lattice_string = "lattice=";
    if (tokens[n].substr(0, lattice_string.length()) == lattice_string) {
      has_lattice_in_exyz = true;
      const int transpose_index[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
      for (int m = 0; m < 9; ++m) {
        box.cpu_h[transpose_index[m]] = get_double_from_token(
          tokens[n + m].substr(
            (m == 0) ? (lattice_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__,
          __LINE__);
      }
    }
  }
  if (!has_lattice_in_exyz) {
    PRINT_INPUT_ERROR("'lattice' is missing in the second line of the model file.");
  } else {
    printf("Box matrix h = [a, b, c] is\n");
    for (int d1 = 0; d1 < 3; ++d1) {
      for (int d2 = 0; d2 < 3; ++d2) {
        printf("%20.10e", box.cpu_h[d1 * 3 + d2]);
      }
      printf("\n");
    }

    box.get_inverse();

    printf("Inverse box matrix g = inv(h) is\n");
    for (int d1 = 0; d1 < 3; ++d1) {
      for (int d2 = 0; d2 < 3; ++d2) {
        printf("%20.10e", box.cpu_h[9 + d1 * 3 + d2]);
      }
      printf("\n");
    }
  }

  // properties
  const int num_properties = 6;
  std::string property_name[num_properties] = {"species", "pos", "mass", "charge", "vel", "group"};
  int property_position[num_properties] = {
    -1, -1, -1, -1, -1, -1}; // species,pos,mass,charge,vel,group
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string properties_string = "properties=";
    if (tokens[n].substr(0, properties_string.length()) == properties_string) {
      std::string line = tokens[n].substr(properties_string.length(), tokens[n].length());
      for (auto& letter : line) {
        if (letter == ':') {
          letter = ' ';
        }
      }
      std::vector<std::string> sub_tokens = get_tokens(line);
      for (int k = 0; k < sub_tokens.size() / 3; ++k) {
        for (int prop = 0; prop < num_properties; ++prop) {
          if (sub_tokens[k * 3] == property_name[prop]) {
            property_position[prop] = k;
          }
        }
      }

      if (property_position[4] < 0) {
        has_velocity_in_xyz = 0;
        printf("Do not specify initial velocities here.\n");
      } else {
        has_velocity_in_xyz = 1;
        printf("Specify initial velocities here.\n");
      }

      if (property_position[5] < 0) {
        group.resize(0);
        printf("Have no grouping method.\n");
      } else {
        int num_of_grouping_methods =
          get_int_from_token(sub_tokens[property_position[5] * 3 + 2], __FILE__, __LINE__);
        group.resize(num_of_grouping_methods);
        printf("Have %d grouping method(s).\n", num_of_grouping_methods);
      }

      for (int k = 0; k < sub_tokens.size() / 3; ++k) {
        const int tmp_length = get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
        for (int prop = 0; prop < num_properties; ++prop) {
          if (k < property_position[prop]) {
            property_offset[prop] += tmp_length;
          }
        }
        num_columns += tmp_length;
      }
    }
  }

  if (property_position[0] < 0) {
    PRINT_INPUT_ERROR("'species' or 'properties' is missing in the model file.");
  }
  if (property_position[1] < 0) {
    PRINT_INPUT_ERROR("'pos' or 'properties' is missing in the model file.");
  }
  if (property_position[2] < 0) {
    has_mass = false;
  } else {
    has_mass = true;
  }
  if (property_position[3] < 0) {
    has_charge = false;
  } else {
    has_charge = true;
  }
}

void read_xyz_in_line_3(
  std::ifstream& input,
  const int N,
  const int has_velocity_in_xyz,
  const bool has_mass,
  const bool has_charge,
  const int num_columns,
  const int* property_offset,
  int& number_of_types,
  std::vector<std::string>& atom_symbols,
  std::vector<std::string>& cpu_atom_symbol,
  std::vector<int>& cpu_type,
  std::vector<double>& cpu_mass,
  std::vector<float>& cpu_charge,
  std::vector<double>& cpu_position_per_atom,
  std::vector<double>& cpu_velocity_per_atom,
  std::vector<Group>& group)
{
  number_of_types = atom_symbols.size();

  std::vector<std::string> temp_atom_symbol(N);
  std::vector<int> temp_type(N);
  std::vector<double> temp_mass(N);
  std::vector<float> temp_charge(N, 0.0);
  std::vector<double> temp_position_per_atom(N * 3);
  std::vector<double> temp_velocity_per_atom(N * 3);
  std::vector<Group> temp_group(group.size());
  for (int m = 0; m < group.size(); ++m) {
    temp_group[m].cpu_label.resize(N);
    temp_group[m].number = 0;
  }

  for (int n = 0; n < N; n++) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != num_columns) {
      PRINT_INPUT_ERROR("number of columns does not match properties.\n");
    }

    temp_atom_symbol[n] = tokens[property_offset[0]];

    bool is_allowed_element = false;
    for (int t = 0; t < number_of_types; ++t) {
      if (temp_atom_symbol[n] == atom_symbols[t]) {
        temp_type[n] = t;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      PRINT_INPUT_ERROR("There is atom in model.xyz that is not allowed in the used potential.\n");
    }

    for (int d = 0; d < 3; ++d) {
      temp_position_per_atom[n + N * d] =
        get_double_from_token(tokens[property_offset[1] + d], __FILE__, __LINE__);
    }

    if (has_mass) {
      temp_mass[n] = get_double_from_token(tokens[property_offset[2]], __FILE__, __LINE__);
      if (temp_mass[n] <= 0) {
        PRINT_INPUT_ERROR("Atom mass should > 0.");
      }
    } else {
      temp_mass[n] = MASS_TABLE.at(temp_atom_symbol[n]);
    }

    if (has_charge) {
      temp_charge[n] = get_double_from_token(tokens[property_offset[3]], __FILE__, __LINE__);
    }

    if (has_velocity_in_xyz) {
      const double A_per_fs_to_natural = TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        temp_velocity_per_atom[n + N * d] =
          get_double_from_token(tokens[property_offset[4] + d], __FILE__, __LINE__) *
          A_per_fs_to_natural;
      }
    }

    for (int m = 0; m < group.size(); ++m) {
      temp_group[m].cpu_label[n] =
        get_int_from_token(tokens[property_offset[5] + m], __FILE__, __LINE__);
      if (temp_group[m].cpu_label[n] < 0 || temp_group[m].cpu_label[n] >= N) {
        PRINT_INPUT_ERROR("Group label should >= 0 and < N.");
      }
      if ((temp_group[m].cpu_label[n] + 1) > temp_group[m].number) {
        temp_group[m].number = temp_group[m].cpu_label[n] + 1;
      }
    }
  }

  cpu_atom_symbol.resize(N);
  cpu_type.resize(N);
  cpu_mass.resize(N);
  cpu_charge.resize(N, 0.0);
  cpu_position_per_atom.resize(N * 3);
  cpu_velocity_per_atom.resize(N * 3);

  for (int m = 0; m < group.size(); ++m) {
    group[m].cpu_label.resize(N);
    group[m].number = temp_group[m].number;
  }

  sort_position(
    N, 
    temp_position_per_atom.data(), 
    temp_position_per_atom.data() + N, 
    temp_position_per_atom.data() + N * 2, 
    temp_velocity_per_atom.data(), 
    temp_velocity_per_atom.data() + N, 
    temp_velocity_per_atom.data() + N * 2, 
    temp_atom_symbol.data(),
    temp_type.data(),
    temp_mass.data(),
    temp_charge.data(),
    temp_group,
    cpu_position_per_atom.data(), 
    cpu_position_per_atom.data() + N, 
    cpu_position_per_atom.data() + N * 2, 
    cpu_velocity_per_atom.data(), 
    cpu_velocity_per_atom.data() + N, 
    cpu_velocity_per_atom.data() + N * 2, 
    cpu_atom_symbol.data(),
    cpu_type.data(),
    cpu_mass.data(),
    cpu_charge.data(),
    group);
}

void find_type_size(
  const int N,
  const int number_of_types,
  const std::vector<int>& cpu_type,
  std::vector<int>& cpu_type_size)
{
  cpu_type_size.resize(number_of_types);

  if (number_of_types == 1) {
    printf("There is only one atom type.\n");
  } else {
    printf("There are %d atom types.\n", number_of_types);
  }

  for (int m = 0; m < number_of_types; m++) {
    cpu_type_size[m] = 0;
  }
  for (int n = 0; n < N; n++) {
    cpu_type_size[cpu_type[n]]++;
  }
  for (int m = 0; m < number_of_types; m++) {
    printf("    %d atoms of type %d.\n", cpu_type_size[m], m);
  }
}

static std::string get_filename_potential()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("No run.in.");
  }

  std::string line;
  std::string filename_potential;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 2) {
      if (tokens[0] == "potential") {
        filename_potential = tokens[1];
      }
    }
  }
  input_run.close();
  if (filename_potential.size() == 0) {
    PRINT_INPUT_ERROR("There is no 'potential' keyword in run.in.");
  } else {
    return filename_potential;
  }
}

static std::vector<std::string> get_atom_symbols(std::string& filename_potential)
{
  std::ifstream input_potential(filename_potential);
  if (!input_potential.is_open()) {
    std::cout << "Error: cannot open " + filename_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input_potential);
  if (tokens.size() < 3) {
    std::cout << "The first line of the potential file should have at least 3 items." << std::endl;
    exit(1);
  }

  int number_of_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + number_of_types) {
    std::cout << "The first line of the potential file should have " << number_of_types
              << " atom symbols." << std::endl;
    exit(1);
  }

  std::vector<std::string> atom_symbols(number_of_types);
  for (int n = 0; n < number_of_types; ++n) {
    atom_symbols[n] = tokens[2 + n];
  }

  input_potential.close();
  return atom_symbols;
}

void initialize_position(
  int& has_velocity_in_xyz, int& number_of_types, Box& box, std::vector<Group>& group, Atom& atom)
{
  std::string filename("model.xyz");
  std::ifstream input(filename);

  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open model.xyz.");
  }

  std::vector<std::string> atom_symbols;
  auto filename_potential = get_filename_potential();
  atom_symbols = get_atom_symbols(filename_potential);

  read_xyz_line_1(input, atom.number_of_atoms);
  int property_offset[6] = {0, 0, 0, 0, 0, 0}; // species,pos,mass,vel,group
  int num_columns = 0;
  bool has_mass = true;
  bool has_charge = true;
  read_xyz_line_2(
    input, box, has_velocity_in_xyz, has_mass, has_charge, num_columns, property_offset, group);

  read_xyz_in_line_3(
    input,
    atom.number_of_atoms,
    has_velocity_in_xyz,
    has_mass,
    has_charge,
    num_columns,
    property_offset,
    number_of_types,
    atom_symbols,
    atom.cpu_atom_symbol,
    atom.cpu_type,
    atom.cpu_mass,
    atom.cpu_charge,
    atom.cpu_position_per_atom,
    atom.cpu_velocity_per_atom,
    group);

  input.close();

  for (int m = 0; m < group.size(); ++m) {
    group[m].find_size(atom.number_of_atoms, m);
    group[m].find_contents(atom.number_of_atoms);
  }

  find_type_size(atom.number_of_atoms, number_of_types, atom.cpu_type, atom.cpu_type_size);
}

void allocate_memory_gpu(std::vector<Group>& group, Atom& atom, GPU_Vector<double>& thermo)
{
  const int N = atom.number_of_atoms;
  atom.type.resize(N);
  atom.type.copy_from_host(atom.cpu_type.data());
  for (int m = 0; m < group.size(); ++m) {
    group[m].label.resize(N);
    group[m].size.resize(group[m].number);
    group[m].size_sum.resize(group[m].number);
    group[m].contents.resize(N);
    group[m].label.copy_from_host(group[m].cpu_label.data());
    group[m].size.copy_from_host(group[m].cpu_size.data());
    group[m].size_sum.copy_from_host(group[m].cpu_size_sum.data());
    group[m].contents.copy_from_host(group[m].cpu_contents.data());
  }
  atom.mass.resize(N);
  atom.mass.copy_from_host(atom.cpu_mass.data());
  atom.charge.resize(N);
  atom.charge.copy_from_host(atom.cpu_charge.data());
  atom.position_per_atom.resize(N * 3);
  atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.resize(N * 3);
  atom.force_per_atom.resize(N * 3, 0);
  atom.virial_per_atom.resize(N * 9);
  atom.potential_per_atom.resize(N);
  thermo.resize(12);
}