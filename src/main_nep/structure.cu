/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "main_nep/common/nep_parameters.cuh"
#include "structure.cuh"
#include "xyz_reader.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <fstream>
#include <iostream>

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

static void change_box(const NEP_Parameters& para, Structure& structure)
{
  float a[3] = {structure.box_original[0], structure.box_original[3], structure.box_original[6]};
  float b[3] = {structure.box_original[1], structure.box_original[4], structure.box_original[7]};
  float c[3] = {structure.box_original[2], structure.box_original[5], structure.box_original[8]};
  float det = get_det(structure.box_original);
  structure.volume = abs(det);
  if (structure.pbc) {
    structure.num_cell[0] =
      int(ceil(2.0f * para.rc_radial_max / (structure.volume / get_area(b, c))));
    structure.num_cell[1] =
      int(ceil(2.0f * para.rc_radial_max / (structure.volume / get_area(c, a))));
    structure.num_cell[2] =
      int(ceil(2.0f * para.rc_radial_max / (structure.volume / get_area(a, b))));
  } else {
    structure.num_cell[0] = 1;
    structure.num_cell[1] = 1;
    structure.num_cell[2] = 1;
  }

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

void NEP_Data_Reader::read_geometry(const XYZ_Record& record, Structure& structure) const
{
  structure.num_atom = record.num_atoms;
  structure.index_in_file = record.index;
  const auto& tokens = record.tokens;
  const auto& xyz_filename = record.filename;
  int line_number = record.comment_line_number;
  structure.weight = 1.0f;
  for (const auto& token : tokens) {
    const std::string weight_string = "weight=";
    if (token.substr(0, weight_string.length()) == weight_string) {
      structure.weight = get_double_from_token(
        token.substr(weight_string.length(), token.length()), xyz_filename.c_str(), line_number);
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
        structure.box_original[transpose_index[m]] = get_double_from_token(
          tokens[n + m].substr(
            (m == 0) ? (lattice_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          xyz_filename.c_str(),
          line_number);
      }
      change_box(para, structure);
    }
  }
  if (!has_lattice_in_exyz) {
    PRINT_INPUT_ERROR("'lattice' is missing in the second line of a frame.");
  }

  int species_offset = record.get_property_offset("species");
  int pos_offset = record.get_property_offset("pos");
  if (species_offset < 0) {
    PRINT_INPUT_ERROR("'species' is missing in properties.");
  }
  if (pos_offset < 0) {
    PRINT_INPUT_ERROR("'pos' is missing in properties.");
  }
  structure.type.resize(structure.num_atom);
  structure.x.resize(structure.num_atom);
  structure.y.resize(structure.num_atom);
  structure.z.resize(structure.num_atom);
  structure.fx.resize(structure.num_atom);
  structure.fy.resize(structure.num_atom);
  structure.fz.resize(structure.num_atom);
  for (int na = 0; na < structure.num_atom; ++na) {
    const auto& tokens = record.atoms[na];
    line_number = record.comment_line_number + na + 1;
    std::string atom_symbol(tokens[species_offset]);
    structure.x[na] =
      get_double_from_token(tokens[0 + pos_offset], xyz_filename.c_str(), line_number);
    structure.y[na] =
      get_double_from_token(tokens[1 + pos_offset], xyz_filename.c_str(), line_number);
    structure.z[na] =
      get_double_from_token(tokens[2 + pos_offset], xyz_filename.c_str(), line_number);
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

void NEP_Data_Reader::read_energy(
  const XYZ_Record& record, Structure& structure, bool required) const
{
  const auto& tokens = record.tokens;
  const auto& xyz_filename = record.filename;
  int line_number = record.comment_line_number;
  // get energy_weight (optional)
  for (const auto& token : tokens) {
    const std::string energy_weight_string = "energy_weight=";
    if (token.substr(0, energy_weight_string.length()) == energy_weight_string) {
      structure.energy_weight = get_double_from_token(
        token.substr(energy_weight_string.length(), token.length()), xyz_filename.c_str(), line_number);
    }
  }

  bool has_energy_in_exyz = false;
  for (const auto& token : tokens) {
    const std::string energy_string = "energy=";
    if (token.substr(0, energy_string.length()) == energy_string) {
      has_energy_in_exyz = true;
      structure.energy_total = get_double_from_token(
        token.substr(energy_string.length(), token.length()), xyz_filename.c_str(), line_number);
      structure.energy = structure.energy_total / structure.num_atom;
    }
  }
  if (required && !has_energy_in_exyz) {
    PRINT_INPUT_ERROR("'energy' is missing in the second line of a frame.");
  }
}

void NEP_Data_Reader::read_structure(const XYZ_Record& record, Structure& structure)
{
  read_geometry(record, structure);
  read_energy(record, structure, true);
  const auto& tokens = record.tokens;
  const auto& xyz_filename = record.filename;
  int line_number = record.comment_line_number;
  structure.has_virial = false;
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string virial_string = "virial=";
    if (tokens[n].substr(0, virial_string.length()) == virial_string) {
      structure.has_virial = true;
      const int reduced_index[9] = {0, 3, 5, 3, 1, 4, 5, 4, 2};
      for (int m = 0; m < 9; ++m) {
        structure.virial[reduced_index[m]] = get_double_from_token(
          tokens[n + m].substr(
            (m == 0) ? (virial_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          xyz_filename.c_str(),
          line_number);
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
        virials_from_stress[reduced_index[m]] = get_double_from_token(
          tokens[n + m].substr(
            (m == 0) ? (stress_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          xyz_filename.c_str(),
          line_number);
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

  int force_offset = record.get_property_offset("force", "forces");
  if (force_offset < 0) {
    PRINT_INPUT_ERROR("'force' or 'forces' is missing in properties.");
  }
  for (int na = 0; na < structure.num_atom; ++na) {
    const auto& tokens = record.atoms[na];
    line_number = record.comment_line_number + na + 1;
    structure.fx[na] =
      get_double_from_token(tokens[0 + force_offset], xyz_filename.c_str(), line_number);
    structure.fy[na] =
      get_double_from_token(tokens[1 + force_offset], xyz_filename.c_str(), line_number);
    structure.fz[na] =
      get_double_from_token(tokens[2 + force_offset], xyz_filename.c_str(), line_number);
  }
}

void read_nep_boundaries(const XYZ_Record& record, Structure& structure)
{
  const auto& tokens = record.tokens;
  // Read boundaries for long-range models. Keep PPP as the default for
  // existing training data.
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string pbc_string = "pbc=";
    if (tokens[n].substr(0, pbc_string.length()) == pbc_string) {
      if (n + 2 >= tokens.size()) {
        PRINT_INPUT_ERROR("The pbc field should contain three values.");
      }
      const char pbc[3] = {
        tokens[n].back(), tokens[n + 1].front(), tokens[n + 2].front()};
      for (int d = 0; d < 3; ++d) {
        if (pbc[d] != 't' && pbc[d] != 'f') {
          PRINT_INPUT_ERROR("Each pbc value should be T or F.");
        }
      }
      if (pbc[0] != pbc[1] || pbc[1] != pbc[2]) {
        PRINT_INPUT_ERROR("Long-range models support only pbc=\"T T T\" or pbc=\"F F F\".");
      }
      structure.pbc = (pbc[0] == 't');
    }
  }
}

void print_nep_boundaries(const std::vector<Structure>& structures)
{
  int Nc = structures.size();
  int num_fff = 0;
  for (const auto& structure : structures) {
    num_fff += 1 - structure.pbc;
  }
  printf("Number of PPP configurations = %d.\n", Nc - num_fff);
  printf("Number of FFF configurations = %d.\n", num_fff);
}

void NEP_Data_Reader::print_summary(const std::vector<Structure>& structures) const
{
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

bool NEP_Data_Reader::read_structures(bool is_train, std::vector<Structure>& structures)
{
  begin_reading();
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
    std::string xyz_filename = is_train ? "train.xyz" : "test.xyz";
    std::cout << "Started reading " << xyz_filename << std::endl;
    print_line_2();
    XYZ_Reader reader(input, xyz_filename);
    XYZ_Record record;
    while (reader.read(record)) {
      Structure structure;
      read_structure(record, structure);
      structures.emplace_back(std::move(structure));
    }
    printf("Number of configurations = %d.\n", (int)structures.size());
    print_summary(structures);
    input.close();
  }

  return has_test_set;
}
