/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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
  int& num_columns,
  int* property_offset,
  std::vector<Group>& group)
{
  std::vector<std::string> tokens = get_tokens(input);
  for (auto& token : tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }

  box.triclinic = 1; // default is triclinic
  for (const auto& token : tokens) {
    const std::string tmp_string = "triclinic=";
    if (token.substr(0, tmp_string.length()) == tmp_string) {
      if (token.back() == 't') {
        box.triclinic = 1;
      } else if (token.back() == 'f') {
        box.triclinic = 0;
      } else {
        PRINT_INPUT_ERROR("tricinic should be T or F.");
      }
    }
  }
  (box.triclinic == 0) ? printf("Use orthogonal box.\n") : printf("Use triclinic box.\n");

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
          __FILE__, __LINE__);
      }
      box.get_inverse();
      if (!box.triclinic) {
        box.cpu_h[1] = box.cpu_h[4];
        box.cpu_h[2] = box.cpu_h[8];
        box.cpu_h[3] = box.cpu_h[0] * 0.5;
        box.cpu_h[4] = box.cpu_h[1] * 0.5;
        box.cpu_h[5] = box.cpu_h[2] * 0.5;
      }
    }
  }
  if (!has_lattice_in_exyz) {
    PRINT_INPUT_ERROR("'lattice' is missing in the second line of the model file.");
  } else {
    if (box.triclinic == 1) {
      printf("Box matrix h = [a, b, c] is\n");
      for (int d1 = 0; d1 < 3; ++d1) {
        for (int d2 = 0; d2 < 3; ++d2) {
          printf("%20.10e", box.cpu_h[d1 * 3 + d2]);
        }
        printf("\n");
      }

      printf("Inverse box matrix g = inv(h) is\n");
      for (int d1 = 0; d1 < 3; ++d1) {
        for (int d2 = 0; d2 < 3; ++d2) {
          printf("%20.10e", box.cpu_h[9 + d1 * 3 + d2]);
        }
        printf("\n");
      }
    } else {
      if (box.cpu_h[0] <= 0) {
        PRINT_INPUT_ERROR("Box length in x direction <= 0.");
      }
      if (box.cpu_h[1] <= 0) {
        PRINT_INPUT_ERROR("Box length in y direction <= 0.");
      }
      if (box.cpu_h[2] <= 0) {
        PRINT_INPUT_ERROR("Box length in z direction <= 0.");
      }
      printf("Box lengths are\n");
      printf("    Lx = %20.10e A\n", box.cpu_h[0]);
      printf("    Ly = %20.10e A\n", box.cpu_h[1]);
      printf("    Lz = %20.10e A\n", box.cpu_h[2]);
    }
  }

  // properties
  std::string property_name[5] = {"species", "pos", "mass", "vel", "group"};
  int property_position[5] = {-1, -1, -1, -1, -1}; // species,pos,mass,vel,group
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
        for (int prop = 0; prop < 5; ++prop) {
          if (prop == 0) {
            if (sub_tokens[k * 3] == property_name[prop] || sub_tokens[k * 3] == "numbers") {
              property_position[prop] = k;
            }
          } else {
            if (sub_tokens[k * 3] == property_name[prop]) {
              property_position[prop] = k;
            }
          }
        }
      }

      if (property_position[3] < 0) {
        has_velocity_in_xyz = 0;
        printf("Do not specify initial velocities here.\n");
      } else {
        has_velocity_in_xyz = 1;
        printf("Specify initial velocities here.\n");
      }

      if (property_position[4] < 0) {
        group.resize(0);
        printf("Have no grouping method.\n");
      } else {
        int num_of_grouping_methods =
          get_int_from_token(sub_tokens[property_position[4] * 3 + 2], __FILE__, __LINE__);
        group.resize(num_of_grouping_methods);
        printf("Have %d grouping method(s).\n", num_of_grouping_methods);
      }

      for (int k = 0; k < sub_tokens.size() / 3; ++k) {
        const int tmp_length = get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
        for (int prop = 0; prop < 5; ++prop) {
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
    PRINT_INPUT_ERROR("'mass' or 'properties' is missing in the model file.");
  }
}

void read_xyz_in_line_3(
  std::ifstream& input,
  const int N,
  const int has_velocity_in_xyz,
  const int num_columns,
  const int* property_offset,
  int& number_of_types,
  std::vector<std::string>& atom_symbols,
  std::vector<std::string>& cpu_atom_symbol,
  std::vector<int>& cpu_type,
  std::vector<double>& cpu_mass,
  std::vector<double>& cpu_position_per_atom,
  std::vector<double>& cpu_velocity_per_atom,
  std::vector<Group>& group)
{
  cpu_atom_symbol.resize(N);
  cpu_type.resize(N);
  cpu_mass.resize(N);
  cpu_position_per_atom.resize(N * 3);
  cpu_velocity_per_atom.resize(N * 3);
#ifdef USE_NEP
  number_of_types = atom_symbols.size();
#else
  number_of_types = 0;
#endif

  for (int m = 0; m < group.size(); ++m) {
    group[m].cpu_label.resize(N);
    group[m].number = 0;
  }

  for (int n = 0; n < N; n++) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != num_columns) {
      PRINT_INPUT_ERROR("number of columns does not match properties.\n");
    }

    cpu_atom_symbol[n] = tokens[property_offset[0]];

#ifdef USE_NEP
    bool is_allowed_element = false;
    for (int t = 0; t < number_of_types; ++t) {
      if (cpu_atom_symbol[n] == atom_symbols[t]) {
        cpu_type[n] = t;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      PRINT_INPUT_ERROR("There is atom in xyz.in that is not allowed in the used NEP potential.\n");
    }
#else
    cpu_type[n] = get_int_from_token(tokens[property_offset[0]], __FILE__, __LINE__);
#endif

#ifndef USE_NEP
    if (cpu_type[n] < 0 || cpu_type[n] >= N) {
      PRINT_INPUT_ERROR("Atom type should >= 0 and < N.");
    }
    if ((cpu_type[n] + 1) > number_of_types) {
      number_of_types = cpu_type[n] + 1;
    }
#endif

    for (int d = 0; d < 3; ++d) {
      cpu_position_per_atom[n + N * d] =
        get_double_from_token(tokens[property_offset[1] + d], __FILE__, __LINE__);
    }

    cpu_mass[n] = get_double_from_token(tokens[property_offset[2]], __FILE__, __LINE__);
    if (cpu_mass[n] <= 0) {
      PRINT_INPUT_ERROR("Atom mass should > 0.");
    }

    if (has_velocity_in_xyz) {
      for (int d = 0; d < 3; ++d) {
        cpu_velocity_per_atom[n + N * d] =
          get_double_from_token(tokens[property_offset[3] + d], __FILE__, __LINE__);
      }
    }

    for (int m = 0; m < group.size(); ++m) {
      group[m].cpu_label[n] =
        get_int_from_token(tokens[property_offset[4] + m], __FILE__, __LINE__);
      if (group[m].cpu_label[n] < 0 || group[m].cpu_label[n] >= N) {
        PRINT_INPUT_ERROR("Group label should >= 0 and < N.");
      }
      if ((group[m].cpu_label[n] + 1) > group[m].number) {
        group[m].number = group[m].cpu_label[n] + 1;
      }
    }
  }
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

static std::string get_filename_potential(char* input_dir)
{
  std::ifstream input_run(input_dir + std::string("/run.in"));
  if (!input_run.is_open()) {
    input_run.open(input_dir + std::string("/phonon.in"));
    if (!input_run.is_open()) {
      PRINT_INPUT_ERROR("No run.in or phonon.in.");
    }
  }

  std::string line;
  std::string filename_potential;
  while (std::getline(input_run, line)) {
    std::stringstream ss(line);
    std::string token;
    ss >> token;
    if (token == "potential") {
      ss >> filename_potential;
    }
  }
  input_run.close();

  return filename_potential;
}

static int get_potential_type(std::string& filename_potential)
{
  std::ifstream input_potential(filename_potential);
  if (!input_potential.is_open()) {
    std::cout << "Error: cannot open " + filename_potential << std::endl;
    exit(1);
  }

  std::string potential_name;
  input_potential >> potential_name;
  input_potential.close();

  if (potential_name == "fcp") {
    return 1;
  } else if (potential_name.substr(0, 3) == "nep") {
    return 2;
  } else {
    return 0; // empirical potentials
  }
}

#ifdef USE_NEP
static std::vector<std::string> get_atom_symbols(std::string& filename_potential)
{
  std::ifstream input_potential(filename_potential);
  if (!input_potential.is_open()) {
    std::cout << "Error: cannot open " + filename_potential << std::endl;
    exit(1);
  }

  std::string potential_name;
  input_potential >> potential_name;
  if (potential_name.substr(0, 3) != "nep") {
    PRINT_INPUT_ERROR(
      "Error: The potential name must be started with 'nep' in this compiled version.");
    exit(1);
  }

  int number_of_types;
  input_potential >> number_of_types;
  std::vector<std::string> atom_symbols(number_of_types);
  for (int n = 0; n < number_of_types; ++n) {
    input_potential >> atom_symbols[n];
  }

  input_potential.close();
  return atom_symbols;
}
#endif

void initialize_position(
  char* input_dir,
  int& N,
  int& has_velocity_in_xyz,
  int& number_of_types,
  Box& box,
  std::vector<Group>& group,
  Atom& atom)
{
  std::string filename(input_dir + std::string("/model.xyz"));
  std::ifstream input(filename);

  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open model.xyz.");
  }

  read_xyz_line_1(input, N);
  int property_offset[5] = {0, 0, 0, 0, 0}; // species,pos,mass,vel,group
  int num_columns = 0;
  read_xyz_line_2(input, box, has_velocity_in_xyz, num_columns, property_offset, group);

  std::vector<std::string> atom_symbols;
  auto filename_potential = get_filename_potential(input_dir);

#ifndef USE_NEP
  if (get_potential_type(filename_potential) == 2) {
    PRINT_INPUT_ERROR("You are using NEP potential without adding -DUSE_NEP in makefile.");
  }
#endif

#ifndef USE_FCP
  if (get_potential_type(filename_potential) == 1) {
    PRINT_INPUT_ERROR("You are using FCP potential without adding -DUSE_FCP in makefile.");
  }
#endif

#if defined(USE_FCP) || defined(USE_NEP)
  if (get_potential_type(filename_potential) == 0) {
    PRINT_INPUT_ERROR("You are using empirical potential with -DUSE_FCP or -DUSE_NEP in makefile.");
  }
#endif

#ifdef USE_NEP
  atom_symbols = get_atom_symbols(filename_potential);
#endif

  read_xyz_in_line_3(
    input, N, has_velocity_in_xyz, num_columns, property_offset, number_of_types, atom_symbols,
    atom.cpu_atom_symbol, atom.cpu_type, atom.cpu_mass, atom.cpu_position_per_atom,
    atom.cpu_velocity_per_atom, group);

  input.close();

  for (int m = 0; m < group.size(); ++m) {
    group[m].find_size(N, m);
    group[m].find_contents(N);
  }

  find_type_size(N, number_of_types, atom.cpu_type, atom.cpu_type_size);
}

void allocate_memory_gpu(
  const int N, std::vector<Group>& group, Atom& atom, GPU_Vector<double>& thermo)
{
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
  atom.position_per_atom.resize(N * 3);
  atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.resize(N * 3);
  atom.force_per_atom.resize(N * 3);
  atom.virial_per_atom.resize(N * 9);
  atom.potential_per_atom.resize(N);
  atom.heat_per_atom.resize(N * 5);
  thermo.resize(12);
}