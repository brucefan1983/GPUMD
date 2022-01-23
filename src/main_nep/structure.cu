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

#include "parameters.cuh"
#include "structure.cuh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> get_tokens(std::ifstream& input)
{
  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

static int get_int_from_token(std::string& token, const char* filename, const int line)
{
  int value = 0;
  try {
    value = std::stoi(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

static float get_float_from_token(std::string& token, const char* filename, const int line)
{
  float value = 0;
  try {
    value = std::stof(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

static void read_Nc(std::ifstream& input, std::vector<Structure>& structures)
{
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 1) {
    PRINT_INPUT_ERROR("The first line in trian.in/test.in should have one value.");
  }
  int Nc = get_int_from_token(tokens[0], __FILE__, __LINE__);
  if (Nc < 1) {
    PRINT_INPUT_ERROR("Number of configurations should >= 1.");
  }
  printf("Number of configurations = %d.\n", Nc);
  structures.resize(Nc);
}

static void read_Na(std::ifstream& input, std::vector<Structure>& structures)
{
  for (int nc = 0; nc < structures.size(); ++nc) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() < 2 || tokens.size() > 3) {
      PRINT_INPUT_ERROR("Number of items here must be 2 or 3.");
    } else {
      structures[nc].num_atom = get_int_from_token(tokens[0], __FILE__, __LINE__);
      structures[nc].has_virial = get_int_from_token(tokens[1], __FILE__, __LINE__);
      if (tokens.size() == 3) {
        structures[nc].weight = get_float_from_token(tokens[2], __FILE__, __LINE__);
        if (structures[nc].weight <= 0.0f || structures[nc].weight > 100.0f) {
          PRINT_INPUT_ERROR("Configuration weight should > 0 and <= 100.");
        }
      } else {
        structures[nc].weight = 1.0f; // default weight is 1
      }
    }
    if (structures[nc].num_atom < 1) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should >= 1.");
    }
  }
}

static void read_energy_virial(std::ifstream& input, int nc, std::vector<Structure>& structures)
{
  std::vector<std::string> tokens = get_tokens(input);
  if (structures[nc].has_virial) {
    if (tokens.size() != 7) {
      PRINT_INPUT_ERROR("Number of items here must be 7.");
    }
    structures[nc].energy = get_float_from_token(tokens[0], __FILE__, __LINE__);
    for (int k = 0; k < 6; ++k) {
      structures[nc].virial[k] = get_float_from_token(tokens[k + 1], __FILE__, __LINE__);
      structures[nc].virial[k] /= structures[nc].num_atom;
    }
  } else {

    if (tokens.size() != 1) {
      PRINT_INPUT_ERROR("Number of items here must be 1.");
    }
    structures[nc].energy = get_float_from_token(tokens[0], __FILE__, __LINE__);
  }

  structures[nc].energy /= structures[nc].num_atom;
}

static float get_area(const float* a, const float* b)
{
  float s1 = a[1] * b[2] - a[2] * b[1];
  float s2 = a[2] * b[0] - a[0] * b[2];
  float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

static float get_det(const float* box)
{
  return box[0] * (box[4] * box[8] - box[5] * box[7]) +
         box[1] * (box[5] * box[6] - box[3] * box[8]) +
         box[2] * (box[3] * box[7] - box[4] * box[6]);
}

static void
read_box(std::ifstream& input, int nc, Parameters& para, std::vector<Structure>& structures)
{
  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 9) {
    PRINT_INPUT_ERROR("Number of items for box line must be 9.");
  }

  float a[3], b[3], c[3];
  a[0] = get_float_from_token(tokens[0], __FILE__, __LINE__);
  a[1] = get_float_from_token(tokens[1], __FILE__, __LINE__);
  a[2] = get_float_from_token(tokens[2], __FILE__, __LINE__);
  b[0] = get_float_from_token(tokens[3], __FILE__, __LINE__);
  b[1] = get_float_from_token(tokens[4], __FILE__, __LINE__);
  b[2] = get_float_from_token(tokens[5], __FILE__, __LINE__);
  c[0] = get_float_from_token(tokens[6], __FILE__, __LINE__);
  c[1] = get_float_from_token(tokens[7], __FILE__, __LINE__);
  c[2] = get_float_from_token(tokens[8], __FILE__, __LINE__);

  structures[nc].box_original[0] = a[0];
  structures[nc].box_original[3] = a[1];
  structures[nc].box_original[6] = a[2];
  structures[nc].box_original[1] = b[0];
  structures[nc].box_original[4] = b[1];
  structures[nc].box_original[7] = b[2];
  structures[nc].box_original[2] = c[0];
  structures[nc].box_original[5] = c[1];
  structures[nc].box_original[8] = c[2];

  float det = get_det(structures[nc].box_original);
  float volume = abs(det);
  structures[nc].num_cell[0] = int(ceil(2.0f * para.rc_radial / (volume / get_area(b, c))));
  structures[nc].num_cell[1] = int(ceil(2.0f * para.rc_radial / (volume / get_area(c, a))));
  structures[nc].num_cell[2] = int(ceil(2.0f * para.rc_radial / (volume / get_area(a, b))));

  structures[nc].box[0] = structures[nc].box_original[0] * structures[nc].num_cell[0];
  structures[nc].box[3] = structures[nc].box_original[3] * structures[nc].num_cell[0];
  structures[nc].box[6] = structures[nc].box_original[6] * structures[nc].num_cell[0];
  structures[nc].box[1] = structures[nc].box_original[1] * structures[nc].num_cell[1];
  structures[nc].box[4] = structures[nc].box_original[4] * structures[nc].num_cell[1];
  structures[nc].box[7] = structures[nc].box_original[7] * structures[nc].num_cell[1];
  structures[nc].box[2] = structures[nc].box_original[2] * structures[nc].num_cell[2];
  structures[nc].box[5] = structures[nc].box_original[5] * structures[nc].num_cell[2];
  structures[nc].box[8] = structures[nc].box_original[8] * structures[nc].num_cell[2];

  structures[nc].box[9] =
    structures[nc].box[4] * structures[nc].box[8] - structures[nc].box[5] * structures[nc].box[7];
  structures[nc].box[10] =
    structures[nc].box[2] * structures[nc].box[7] - structures[nc].box[1] * structures[nc].box[8];
  structures[nc].box[11] =
    structures[nc].box[1] * structures[nc].box[5] - structures[nc].box[2] * structures[nc].box[4];
  structures[nc].box[12] =
    structures[nc].box[5] * structures[nc].box[6] - structures[nc].box[3] * structures[nc].box[8];
  structures[nc].box[13] =
    structures[nc].box[0] * structures[nc].box[8] - structures[nc].box[2] * structures[nc].box[6];
  structures[nc].box[14] =
    structures[nc].box[2] * structures[nc].box[3] - structures[nc].box[0] * structures[nc].box[5];
  structures[nc].box[15] =
    structures[nc].box[3] * structures[nc].box[7] - structures[nc].box[4] * structures[nc].box[6];
  structures[nc].box[16] =
    structures[nc].box[1] * structures[nc].box[6] - structures[nc].box[0] * structures[nc].box[7];
  structures[nc].box[17] =
    structures[nc].box[0] * structures[nc].box[4] - structures[nc].box[1] * structures[nc].box[3];

  det *= structures[nc].num_cell[0] * structures[nc].num_cell[1] * structures[nc].num_cell[2];
  for (int n = 9; n < 18; n++) {
    structures[nc].box[n] /= det;
  }
}

static void
read_force(std::ifstream& input, int nc, Parameters& para, std::vector<Structure>& structures)
{
  structures[nc].type.resize(structures[nc].num_atom);
  structures[nc].x.resize(structures[nc].num_atom);
  structures[nc].y.resize(structures[nc].num_atom);
  structures[nc].z.resize(structures[nc].num_atom);
  structures[nc].fx.resize(structures[nc].num_atom);
  structures[nc].fy.resize(structures[nc].num_atom);
  structures[nc].fz.resize(structures[nc].num_atom);

  for (int na = 0; na < structures[nc].num_atom; ++na) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != 7) {
      PRINT_INPUT_ERROR("Number of items for atom line must be 7.");
    }
    std::string atom_symbol(tokens[0]);
    structures[nc].x[na] = get_float_from_token(tokens[1], __FILE__, __LINE__);
    structures[nc].y[na] = get_float_from_token(tokens[2], __FILE__, __LINE__);
    structures[nc].z[na] = get_float_from_token(tokens[3], __FILE__, __LINE__);
    structures[nc].fx[na] = get_float_from_token(tokens[4], __FILE__, __LINE__);
    structures[nc].fy[na] = get_float_from_token(tokens[5], __FILE__, __LINE__);
    structures[nc].fz[na] = get_float_from_token(tokens[6], __FILE__, __LINE__);

    bool is_allowed_element = false;
    for (int n = 0; n < para.elements.size(); ++n) {
      if (atom_symbol == para.elements[n]) {
        structures[nc].type[na] = n;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      PRINT_INPUT_ERROR("There is atom in train.in or test.in that are not in nep.in.\n");
    }
  }
}

static void find_permuted_indices(std::vector<int>& permuted_indices)
{
  std::mt19937 rng;
#ifdef DEBUG
  rng = std::mt19937(54321);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
  for (int i = 0; i < permuted_indices.size(); ++i) {
    permuted_indices[i] = i;
  }
  std::uniform_int_distribution<int> rand_int(0, INT_MAX);
  for (int i = 0; i < permuted_indices.size(); ++i) {
    int j = rand_int(rng) % (permuted_indices.size() - i) + i;
    int temp = permuted_indices[i];
    permuted_indices[i] = permuted_indices[j];
    permuted_indices[j] = temp;
  }
}

static void reorder(std::vector<Structure>& structures)
{
  std::vector<int> configuration_id(structures.size());
  find_permuted_indices(configuration_id);

  std::vector<Structure> structures_copy(structures.size());

  for (int nc = 0; nc < structures.size(); ++nc) {
    structures_copy[nc].num_atom = structures[nc].num_atom;
    structures_copy[nc].weight = structures[nc].weight;
    structures_copy[nc].has_virial = structures[nc].has_virial;
    structures_copy[nc].energy = structures[nc].energy;
    for (int k = 0; k < 6; ++k) {
      structures_copy[nc].virial[k] = structures[nc].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures_copy[nc].box[k] = structures[nc].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      structures_copy[nc].box_original[k] = structures[nc].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      structures_copy[nc].num_cell[k] = structures[nc].num_cell[k];
    }
    structures_copy[nc].type.resize(structures[nc].num_atom);
    structures_copy[nc].x.resize(structures[nc].num_atom);
    structures_copy[nc].y.resize(structures[nc].num_atom);
    structures_copy[nc].z.resize(structures[nc].num_atom);
    structures_copy[nc].fx.resize(structures[nc].num_atom);
    structures_copy[nc].fy.resize(structures[nc].num_atom);
    structures_copy[nc].fz.resize(structures[nc].num_atom);
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      structures_copy[nc].type[na] = structures[nc].type[na];
      structures_copy[nc].x[na] = structures[nc].x[na];
      structures_copy[nc].y[na] = structures[nc].y[na];
      structures_copy[nc].z[na] = structures[nc].z[na];
      structures_copy[nc].fx[na] = structures[nc].fx[na];
      structures_copy[nc].fy[na] = structures[nc].fy[na];
      structures_copy[nc].fz[na] = structures[nc].fz[na];
    }
  }

  for (int nc = 0; nc < structures.size(); ++nc) {
    structures[nc].num_atom = structures_copy[configuration_id[nc]].num_atom;
    structures[nc].weight = structures_copy[configuration_id[nc]].weight;
    structures[nc].has_virial = structures_copy[configuration_id[nc]].has_virial;
    structures[nc].energy = structures_copy[configuration_id[nc]].energy;
    for (int k = 0; k < 6; ++k) {
      structures[nc].virial[k] = structures_copy[configuration_id[nc]].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures[nc].box[k] = structures_copy[configuration_id[nc]].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      structures[nc].box_original[k] = structures_copy[configuration_id[nc]].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      structures[nc].num_cell[k] = structures_copy[configuration_id[nc]].num_cell[k];
    }
    structures[nc].type.resize(structures[nc].num_atom);
    structures[nc].x.resize(structures[nc].num_atom);
    structures[nc].y.resize(structures[nc].num_atom);
    structures[nc].z.resize(structures[nc].num_atom);
    structures[nc].fx.resize(structures[nc].num_atom);
    structures[nc].fy.resize(structures[nc].num_atom);
    structures[nc].fz.resize(structures[nc].num_atom);
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      structures[nc].type[na] = structures_copy[configuration_id[nc]].type[na];
      structures[nc].x[na] = structures_copy[configuration_id[nc]].x[na];
      structures[nc].y[na] = structures_copy[configuration_id[nc]].y[na];
      structures[nc].z[na] = structures_copy[configuration_id[nc]].z[na];
      structures[nc].fx[na] = structures_copy[configuration_id[nc]].fx[na];
      structures[nc].fy[na] = structures_copy[configuration_id[nc]].fy[na];
      structures[nc].fz[na] = structures_copy[configuration_id[nc]].fz[na];
    }
  }
}

void read_structures(
  bool is_train, char* input_dir, Parameters& para, std::vector<Structure>& structures)
{
  std::string file_train(input_dir);
  if (is_train) {
    file_train += "/train.in";
  } else {
    file_train += "/test.in";
  }
  std::ifstream input(file_train);

  if (!input.is_open()) {
    PRINT_INPUT_ERROR("Failed to open train.in or test.in.");
  }

  read_Nc(input, structures);
  read_Na(input, structures);
  for (int n = 0; n < structures.size(); ++n) {
    read_energy_virial(input, n, structures);
    read_box(input, n, para, structures);
    read_force(input, n, para, structures);
  }

  input.close();

  // only reorder if not using full batch
  if (is_train && (para.batch_size < structures.size())) {
    reorder(structures);
  }
}
