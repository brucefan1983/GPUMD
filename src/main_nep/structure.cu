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
#include "utilities/error.cuh"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

static void change_box(const Parameters& para, Structure& structure)
{
  float a[3] = {structure.box_original[0], structure.box_original[3], structure.box_original[6]};
  float b[3] = {structure.box_original[1], structure.box_original[4], structure.box_original[7]};
  float c[3] = {structure.box_original[2], structure.box_original[5], structure.box_original[8]};
  float det = get_det(structure.box_original);
  float volume = abs(det);
  structure.num_cell[0] = int(ceil(2.0f * para.rc_radial / (volume / get_area(b, c))));
  structure.num_cell[1] = int(ceil(2.0f * para.rc_radial / (volume / get_area(c, a))));
  structure.num_cell[2] = int(ceil(2.0f * para.rc_radial / (volume / get_area(a, b))));

  structure.box[0] = structure.box_original[0] * structure.num_cell[0];
  structure.box[3] = structure.box_original[3] * structure.num_cell[0];
  structure.box[6] = structure.box_original[6] * structure.num_cell[0];
  structure.box[1] = structure.box_original[1] * structure.num_cell[1];
  structure.box[4] = structure.box_original[4] * structure.num_cell[1];
  structure.box[7] = structure.box_original[7] * structure.num_cell[1];
  structure.box[2] = structure.box_original[2] * structure.num_cell[2];
  structure.box[5] = structure.box_original[5] * structure.num_cell[2];
  structure.box[8] = structure.box_original[8] * structure.num_cell[2];

  structure.box[9] = structure.box[4] * structure.box[8] - structure.box[5] * structure.box[7];
  structure.box[10] = structure.box[2] * structure.box[7] - structure.box[1] * structure.box[8];
  structure.box[11] = structure.box[1] * structure.box[5] - structure.box[2] * structure.box[4];
  structure.box[12] = structure.box[5] * structure.box[6] - structure.box[3] * structure.box[8];
  structure.box[13] = structure.box[0] * structure.box[8] - structure.box[2] * structure.box[6];
  structure.box[14] = structure.box[2] * structure.box[3] - structure.box[0] * structure.box[5];
  structure.box[15] = structure.box[3] * structure.box[7] - structure.box[4] * structure.box[6];
  structure.box[16] = structure.box[1] * structure.box[6] - structure.box[0] * structure.box[7];
  structure.box[17] = structure.box[0] * structure.box[4] - structure.box[1] * structure.box[3];

  det *= structure.num_cell[0] * structure.num_cell[1] * structure.num_cell[2];
  for (int n = 9; n < 18; n++) {
    structure.box[n] /= det;
  }
}

static void read_force(
  const int num_columns,
  const int species_offset,
  const int pos_offset,
  const int force_offset,
  std::ifstream& input,
  const Parameters& para,
  Structure& structure)
{
  structure.type.resize(structure.num_atom);
  structure.x.resize(structure.num_atom);
  structure.y.resize(structure.num_atom);
  structure.z.resize(structure.num_atom);
  structure.fx.resize(structure.num_atom);
  structure.fy.resize(structure.num_atom);
  structure.fz.resize(structure.num_atom);

  for (int na = 0; na < structure.num_atom; ++na) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != num_columns) {
      PRINT_INPUT_ERROR("Number of items for an atom line mismatches properties.");
    }
    std::string atom_symbol(tokens[0 + species_offset]);
    structure.x[na] = get_float_from_token(tokens[0 + pos_offset], __FILE__, __LINE__);
    structure.y[na] = get_float_from_token(tokens[1 + pos_offset], __FILE__, __LINE__);
    structure.z[na] = get_float_from_token(tokens[2 + pos_offset], __FILE__, __LINE__);
    if (num_columns > 4) {
      structure.fx[na] = get_float_from_token(tokens[0 + force_offset], __FILE__, __LINE__);
      structure.fy[na] = get_float_from_token(tokens[1 + force_offset], __FILE__, __LINE__);
      structure.fz[na] = get_float_from_token(tokens[2 + force_offset], __FILE__, __LINE__);
    }

    bool is_allowed_element = false;
    for (int n = 0; n < para.elements.size(); ++n) {
      if (atom_symbol == para.elements[n]) {
        structure.type[na] = n;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      PRINT_INPUT_ERROR("There is atom in train.xyz or test.xyz that are not in nep.in.\n");
    }
  }
}

static void read_one_structure(const Parameters& para, std::ifstream& input, Structure& structure)
{
  std::vector<std::string> tokens = get_tokens_without_unwanted_spaces(input);
  for (auto& token : tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }

  if (tokens.size() == 0) {
    PRINT_INPUT_ERROR("The second line for each frame should not be empty.");
  }

  bool has_energy_in_exyz = false;
  for (const auto& token : tokens) {
    const std::string energy_string = "energy=";
    if (token.substr(0, energy_string.length()) == energy_string) {
      has_energy_in_exyz = true;
      structure.energy = get_float_from_token(
        token.substr(energy_string.length(), token.length()), __FILE__, __LINE__);
      structure.energy /= structure.num_atom;
    }
  }
  if (para.train_mode == 0 && !has_energy_in_exyz) {
    PRINT_INPUT_ERROR("'energy' is missing in the second line of a frame.");
  }

  structure.has_temperature = false;
  for (const auto& token : tokens) {
    const std::string temperature_string = "temperature=";
    if (token.substr(0, temperature_string.length()) == temperature_string) {
      structure.has_temperature = true;
      structure.temperature = get_float_from_token(
        token.substr(temperature_string.length(), token.length()), __FILE__, __LINE__);
    }
  }
  if (para.train_mode == 3 && !structure.has_temperature) {
    PRINT_INPUT_ERROR("'temperature' is missing in the second line of a frame.");
  }
  if (!structure.has_temperature) {
    structure.temperature = 0;
  }

  structure.weight = 1.0f;
  for (const auto& token : tokens) {
    const std::string weight_string = "weight=";
    if (token.substr(0, weight_string.length()) == weight_string) {
      structure.weight = get_float_from_token(
        token.substr(weight_string.length(), token.length()), __FILE__, __LINE__);
      if (structure.weight <= 0.0f || structure.weight > 100.0f) {
        PRINT_INPUT_ERROR("Configuration weight should > 0 and <= 100.");
      }
    }
  }

  bool has_lattice_in_exyz = false;
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string lattice_string = "lattice=";
    if (tokens[n].substr(0, lattice_string.length()) == lattice_string) {
      has_lattice_in_exyz = true;
      const int transpose_index[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
      for (int m = 0; m < 9; ++m) {
        structure.box_original[transpose_index[m]] = get_float_from_token(
          tokens[n + m].substr(
            (m == 0) ? (lattice_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__,
          __LINE__);
      }
      change_box(para, structure);
    }
  }
  if (!has_lattice_in_exyz) {
    PRINT_INPUT_ERROR("'lattice' is missing in the second line of a frame.");
  }

  structure.has_virial = false;
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string virial_string = "virial=";
    if (tokens[n].substr(0, virial_string.length()) == virial_string) {
      structure.has_virial = true;
      const int reduced_index[9] = {0, 3, 5, 3, 1, 4, 5, 4, 2};
      for (int m = 0; m < 9; ++m) {
        structure.virial[reduced_index[m]] = get_float_from_token(
          tokens[n + m].substr(
            (m == 0) ? (virial_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__,
          __LINE__);
        structure.virial[reduced_index[m]] /= structure.num_atom;
      }
    }
  }
  // if stresses are available, read them and convert them to per atom virials
  bool has_stress = false;
  std::vector<float> virials_from_stress(6);
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string stress_string = "stress=";
    if (tokens[n].substr(0, stress_string.length()) == stress_string) {
      has_stress = true;
      float volume = abs(get_det(structure.box_original));
      const int reduced_index[9] = {0, 3, 5, 3, 1, 4, 5, 4, 2};
      for (int m = 0; m < 9; ++m) {
        virials_from_stress[reduced_index[m]] = get_float_from_token(
          tokens[n + m].substr(
            (m == 0) ? (stress_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__,
          __LINE__);
        virials_from_stress[reduced_index[m]] *= -volume / structure.num_atom;
      }
    }
  }
  if (structure.has_virial && has_stress) {
    // assert stresses and virials are consistent
    const float tol = 1e-3;
    for (int m = 0; m < 6; ++m) {
      if (abs(structure.virial[m] - virials_from_stress[m]) > tol) {
        if (para.prediction == 0) {
          PRINT_INPUT_ERROR("Virials and stresses for structure are inconsistent!");
        }
      }
    }
    if (para.prediction == 0) {
      std::cout
        << "Structure has both defined virials and stresses. Will use virial information.\n";
    }
  } else if (!structure.has_virial && has_stress) {
    // save virials from stress to structure virials
    for (int m = 0; m < 6; ++m) {
      structure.virial[m] = virials_from_stress[m];
    }
    structure.has_virial = true;
  }
  if (!structure.has_virial) {
    for (int m = 0; m < 6; ++m) {
      structure.virial[m] = -1e6;
    }
  }

  // use the virial viriable to keep the dipole data
  if (para.train_mode == 1) {
    structure.has_virial = false;
    for (int n = 0; n < tokens.size(); ++n) {
      const std::string dipole_string = "dipole=";
      if (tokens[n].substr(0, dipole_string.length()) == dipole_string) {
        structure.has_virial = true;
        for (int m = 0; m < 6; ++m) {
          structure.virial[m] = 0.0f;
        }
        for (int m = 0; m < 3; ++m) {
          structure.virial[m] = get_float_from_token(
            tokens[n + m].substr(
              (m == 0) ? (dipole_string.length() + 1) : 0,
              (m == 2) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
            __FILE__,
            __LINE__);
          structure.virial[m] /= structure.num_atom;
        }
      }
    }
    if (!structure.has_virial) {
      if (para.prediction == 0) {
        PRINT_INPUT_ERROR("'dipole' is missing in the second line of a frame.");
      } else {
        for (int m = 0; m < 6; ++m) {
          structure.virial[m] = -1e6;
        }
      }
    }
  }

  // use the virial viriable to keep the polarizability data
  if (para.train_mode == 2) {
    structure.has_virial = false;
    for (int n = 0; n < tokens.size(); ++n) {
      const std::string pol_string = "pol=";
      if (tokens[n].substr(0, pol_string.length()) == pol_string) {
        structure.has_virial = true;
        const int reduced_index[9] = {0, 3, 5, 3, 1, 4, 5, 4, 2};
        for (int m = 0; m < 9; ++m) {
          structure.virial[reduced_index[m]] = get_float_from_token(
            tokens[n + m].substr(
              (m == 0) ? (pol_string.length() + 1) : 0,
              (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
            __FILE__,
            __LINE__);
          structure.virial[reduced_index[m]] /= structure.num_atom;
        }
      }
    }
    if (!structure.has_virial) {
      if (para.prediction == 0) {
        PRINT_INPUT_ERROR("'pol' is missing in the second line of a frame.");
      } else {
        for (int m = 0; m < 6; ++m) {
          structure.virial[m] = -1e6;
        }
      }
    }
  }

  int species_offset = 0;
  int pos_offset = 0;
  int force_offset = 0;
  int num_columns = 0;
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
      int species_position = -1;
      int pos_position = -1;
      int force_position = -1;
      for (int k = 0; k < sub_tokens.size() / 3; ++k) {
        if (sub_tokens[k * 3] == "species") {
          species_position = k;
        }
        if (sub_tokens[k * 3] == "pos") {
          pos_position = k;
        }
        if (sub_tokens[k * 3] == "force" || sub_tokens[k * 3] == "forces") {
          force_position = k;
        }
      }
      if (species_position < 0) {
        PRINT_INPUT_ERROR("'species' is missing in properties.");
      }
      if (pos_position < 0) {
        PRINT_INPUT_ERROR("'pos' is missing in properties.");
      }
      if (force_position < 0 && para.train_mode == 0) {
        PRINT_INPUT_ERROR("'force' or 'forces' is missing in properties.");
      }
      for (int k = 0; k < sub_tokens.size() / 3; ++k) {
        if (k < species_position) {
          species_offset += get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
        }
        if (k < pos_position) {
          pos_offset += get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
        }
        if (k < force_position) {
          force_offset += get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
        }
        num_columns += get_int_from_token(sub_tokens[k * 3 + 2], __FILE__, __LINE__);
      }
    }
  }

  read_force(num_columns, species_offset, pos_offset, force_offset, input, para, structure);
}

static void
read_exyz(const Parameters& para, std::ifstream& input, std::vector<Structure>& structures)
{
  int Nc = 0;
  while (true) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() == 0) {
      break;
    } else if (tokens.size() > 1) {
      PRINT_INPUT_ERROR("The first line for each frame should have one value.");
    }
    Structure structure;
    structure.num_atom = get_int_from_token(tokens[0], __FILE__, __LINE__);
    if (structure.num_atom < 1) {
      PRINT_INPUT_ERROR("Number of atoms for each frame should >= 1.");
    }
    read_one_structure(para, input, structure);
    structures.emplace_back(structure);
    ++Nc;
  }
  printf("Number of configurations = %d.\n", Nc);

  for (const auto& s : structures) {
    if (s.energy < -100.0f) {
      std::cout << "Warning: \n";
      std::cout << "    There is energy < -100 eV/atom in the data set.\n";
      std::cout << "    Because we use single precision in NEP training\n";
      std::cout << "    it means that the reference and calculated energies\n";
      std::cout << "    might only be accurate up to 1 meV/atom\n";
      std::cout << "    which can effectively introduce noises.\n";
      std::cout << "    We suggest you preprocess (using double precision)\n";
      std::cout << "    your data to make the energies closer to 0." << std::endl;
      break;
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
    structures_copy[nc].has_temperature = structures[nc].has_temperature;
    structures_copy[nc].temperature = structures[nc].temperature;
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
    structures[nc].has_temperature = structures_copy[configuration_id[nc]].has_temperature;
    structures[nc].temperature = structures_copy[configuration_id[nc]].temperature;
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

bool read_structures(bool is_train, Parameters& para, std::vector<Structure>& structures)
{
  std::ifstream input(is_train ? "train.xyz" : "test.xyz");
  bool has_test_set = true;
  if (!input.is_open()) {
    if (is_train) {
      PRINT_INPUT_ERROR("Failed to open train.xyz.");
    } else {
      has_test_set = false;
    }
  } else {
    print_line_1();
    is_train ? printf("Started reading train.xyz.\n") : printf("Started reading test.xyz.\n");
    print_line_2();
    read_exyz(para, input, structures);
    input.close();
  }

  if ((para.prediction == 0) && is_train && (para.batch_size < structures.size())) {
    reorder(structures);
  }

  return has_test_set;
}
