/*-----------------------------------------------------------------------------------------------100
compile:
    g++ -O3 nep_data_toolkit.cpp
run:
    ./a.out
--------------------------------------------------------------------------------------------------*/

#include "nep.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

static std::string remove_spaces_step1(const std::string& line)
{
  std::vector<int> indices_for_spaces(line.size(), 0);
  for (int n = 0; n < line.size(); ++n) {
    if (line[n] == '=') {
      for (int k = 1; n - k >= 0; ++k) {
        if (line[n - k] == ' ' || line[n - k] == '\t') {
          indices_for_spaces[n - k] = 1;
        } else {
          break;
        }
      }
      for (int k = 1; n + k < line.size(); ++k) {
        if (line[n + k] == ' ' || line[n + k] == '\t') {
          indices_for_spaces[n + k] = 1;
        } else {
          break;
        }
      }
    }
  }

  std::string new_line;
  for (int n = 0; n < line.size(); ++n) {
    if (!indices_for_spaces[n]) {
      new_line += line[n];
    }
  }

  return new_line;
}

static std::string remove_spaces(const std::string& line_input)
{
  auto line = remove_spaces_step1(line_input);

  std::vector<int> indices_for_spaces(line.size(), 0);
  for (int n = 0; n < line.size(); ++n) {
    if (line[n] == '\"') {
      if (n == 0) {
        std::cout << "The second line of the .xyz file should not begin with \"." << std::endl;
        exit(1);
      } else {
        if (line[n - 1] == '=') {
          for (int k = 1; n + k < line.size(); ++k) {
            if (line[n + k] == ' ' || line[n + k] == '\t') {
              indices_for_spaces[n + k] = 1;
            } else {
              break;
            }
          }
        } else {
          for (int k = 1; n - k >= 0; ++k) {
            if (line[n - k] == ' ' || line[n - k] == '\t') {
              indices_for_spaces[n - k] = 1;
            } else {
              break;
            }
          }
        }
      }
    }
  }

  std::string new_line;
  for (int n = 0; n < line.size(); ++n) {
    if (!indices_for_spaces[n]) {
      new_line += line[n];
    }
  }

  return new_line;
}

std::vector<std::string> get_tokens(const std::string& line)
{
  std::istringstream iss(line);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

std::vector<std::string> get_tokens_without_unwanted_spaces(std::ifstream& input)
{
  std::string line;
  std::getline(input, line);
  auto line_without_unwanted_spaces = remove_spaces(line);
  std::istringstream iss(line_without_unwanted_spaces);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

std::vector<std::string> get_tokens(std::ifstream& input)
{
  std::string line;
  std::getline(input, line);
  std::istringstream iss(line);
  std::vector<std::string> tokens{
    std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  return tokens;
}

int get_int_from_token(const std::string& token, const char* filename, const int line)
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

double get_double_from_token(const std::string& token, const char* filename, const int line)
{
  double value = 0;
  try {
    value = std::stod(token);
  } catch (const std::exception& e) {
    std::cout << "Standard exception:\n";
    std::cout << "    File:          " << filename << std::endl;
    std::cout << "    Line:          " << line << std::endl;
    std::cout << "    Error message: " << e.what() << std::endl;
    exit(1);
  }
  return value;
}

struct Structure {
  int num_atom;
  std::string sid;
  bool has_sid = false;
  bool has_virial = false;
  bool has_stress = false;
  double charge = 0.0;
  double energy_weight = 1.0;
  double energy;
  double weight;
  double virial[9];
  double stress[9];
  double box[9];
  std::vector<std::string> atom_symbol;
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
  std::vector<double> q;
};

static void read_force(
  const int num_columns,
  const int species_offset,
  const int pos_offset,
  const int force_offset,
  std::ifstream& input,
  Structure& structure)
{
  structure.atom_symbol.resize(structure.num_atom);
  structure.x.resize(structure.num_atom);
  structure.y.resize(structure.num_atom);
  structure.z.resize(structure.num_atom);
  structure.fx.resize(structure.num_atom);
  structure.fy.resize(structure.num_atom);
  structure.fz.resize(structure.num_atom);

  for (int na = 0; na < structure.num_atom; ++na) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != num_columns) {
      std::cout << "Number of items for an atom line mismatches properties." << std::endl;
      exit(1);
    }
    structure.atom_symbol[na] = tokens[0 + species_offset];
    structure.x[na] = get_double_from_token(tokens[0 + pos_offset], __FILE__, __LINE__);
    structure.y[na] = get_double_from_token(tokens[1 + pos_offset], __FILE__, __LINE__);
    structure.z[na] = get_double_from_token(tokens[2 + pos_offset], __FILE__, __LINE__);
    if (num_columns > 4) {
      structure.fx[na] = get_double_from_token(tokens[0 + force_offset], __FILE__, __LINE__);
      structure.fy[na] = get_double_from_token(tokens[1 + force_offset], __FILE__, __LINE__);
      structure.fz[na] = get_double_from_token(tokens[2 + force_offset], __FILE__, __LINE__);
    }
  }
}

static void read_one_structure(std::ifstream& input, Structure& structure)
{
  std::vector<std::string> tokens = get_tokens_without_unwanted_spaces(input);
  for (auto& token : tokens) {
    std::transform(
      token.begin(), token.end(), token.begin(), [](unsigned char c) { return std::tolower(c); });
  }

  if (tokens.size() == 0) {
    std::cout << "The second line for each frame should not be empty." << std::endl;
    exit(1);
  }

  for (const auto& token : tokens) {
    const std::string sid_string = "sid=";
    if (token.substr(0, sid_string.length()) == sid_string) {
      structure.has_sid = true;
      structure.sid = token.substr(sid_string.length(), token.length());
    }
  }

  // get charge (optional)
  for (const auto& token : tokens) {
    const std::string charge_string = "charge=";
    if (token.substr(0, charge_string.length()) == charge_string) {
      structure.charge = get_double_from_token(
        token.substr(charge_string.length(), token.length()), __FILE__, __LINE__);
    }
  }

  // get energy_weight (optional)
  for (const auto& token : tokens) {
    const std::string energy_weight_string = "energy_weight=";
    if (token.substr(0, energy_weight_string.length()) == energy_weight_string) {
      structure.energy_weight = get_double_from_token(
        token.substr(energy_weight_string.length(), token.length()), __FILE__, __LINE__);
    }
  }

  bool has_energy_in_exyz = false;
  for (const auto& token : tokens) {
    const std::string energy_string = "energy=";
    if (token.substr(0, energy_string.length()) == energy_string) {
      has_energy_in_exyz = true;
      structure.energy = get_double_from_token(
        token.substr(energy_string.length(), token.length()), __FILE__, __LINE__);
    }
  }
  if (!has_energy_in_exyz) {
    std::cout << "'energy' is missing in the second line of a frame." << std::endl;
    exit(1);
  }

  structure.weight = 1.0f;
  for (const auto& token : tokens) {
    const std::string weight_string = "weight=";
    if (token.substr(0, weight_string.length()) == weight_string) {
      structure.weight = get_double_from_token(
        token.substr(weight_string.length(), token.length()), __FILE__, __LINE__);
      if (structure.weight <= 0.0f || structure.weight > 100.0f) {
        std::cout << "Configuration weight should > 0 and <= 100." << std::endl;
        exit(1);
      }
    }
  }

  bool has_lattice_in_exyz = false;
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string lattice_string = "lattice=";
    if (tokens[n].substr(0, lattice_string.length()) == lattice_string) {
      has_lattice_in_exyz = true;
      for (int m = 0; m < 9; ++m) {
        structure.box[m] = get_double_from_token(
          tokens[n + m].substr(
            (m == 0) ? (lattice_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__, __LINE__);
      }
    }
  }
  if (!has_lattice_in_exyz) {
    std::cout << "'lattice' is missing in the second line of a frame." << std::endl;
    exit(1);
  }

  structure.has_virial = false;
  for (int n = 0; n < tokens.size(); ++n) {
    const std::string virial_string = "virial=";
    if (tokens[n].substr(0, virial_string.length()) == virial_string) {
      structure.has_virial = true;
      for (int m = 0; m < 9; ++m) {
        structure.virial[m] = get_double_from_token(
          tokens[n + m].substr(
            (m == 0) ? (virial_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__, __LINE__);
      }
    }
  }

  if (!structure.has_virial) {
    for (int n = 0; n < tokens.size(); ++n) {
      const std::string stress_string = "stress=";
      if (tokens[n].substr(0, stress_string.length()) == stress_string) {
        structure.has_stress = true;
        for (int m = 0; m < 9; ++m) {
          structure.stress[m] = get_double_from_token(
            tokens[n + m].substr(
              (m == 0) ? (stress_string.length() + 1) : 0,
              (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
            __FILE__, __LINE__);
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
        std::cout << "'species' is missing in properties." << std::endl;
        exit(1);
      }
      if (pos_position < 0) {
        std::cout << "'pos' is missing in properties." << std::endl;
        exit(1);
      }
      if (force_position < 0) {
        std::cout << "'force' or 'forces' is missing in properties." << std::endl;
        exit(1);
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

  read_force(num_columns, species_offset, pos_offset, force_offset, input, structure);
}

static void read(const std::string& inputfile, std::vector<Structure>& structures)
{
  std::ifstream input(inputfile);
  if (!input.is_open()) {
    std::cout << "Failed to open " << inputfile << std::endl;
    exit(1);
  } else {
    while (true) {
      std::vector<std::string> tokens = get_tokens(input);
      if (tokens.size() == 0) {
        break;
      } else if (tokens.size() > 1) {
        std::cout << "The first line for each frame should have one value." << std::endl;
        exit(1);
      }
      Structure structure;
      structure.num_atom = get_int_from_token(tokens[0], __FILE__, __LINE__);
      if (structure.num_atom < 1) {
        std::cout << "Number of atoms for each frame should >= 1." << std::endl;
        exit(1);
      }
      read_one_structure(input, structure);
      structures.emplace_back(structure);
    }
    input.close();
  }
}

static void write_one_structure(std::ofstream& output, const Structure& structure)
{
  output << structure.num_atom << "\n";
  output << std::fixed << std::setprecision(6);

  if (structure.charge != 0.0) {
    output << "charge=" << structure.charge << " ";
  }

  if (structure.energy_weight != 1.0) {
    output << "energy_weight=" << structure.energy_weight << " ";
  }

  output << "Lattice=\"";
  for (int m = 0; m < 9; ++m) {
    output << structure.box[m];
    if (m != 8) {
      output << " ";
    }
  }
  output << "\" ";

  output << "energy=" << structure.energy << " ";

  if (structure.has_virial) {
    output << "virial=\"";
    for (int m = 0; m < 9; ++m) {
      output << structure.virial[m];
      if (m != 8) {
        output << " ";
      }
    }
    output << "\" ";
  }

  if (structure.has_stress) {
    output << "stress=\"";
    for (int m = 0; m < 9; ++m) {
      output << structure.stress[m];
      if (m != 8) {
        output << " ";
      }
    }
    output << "\" ";
  }

  if (structure.has_sid) {
    output << "sid=" << structure.sid << " ";
  }

  output << "Properties=species:S:1:pos:R:3:force:R:3\n";

  for (int n = 0; n < structure.num_atom; ++n) {
    output << structure.atom_symbol[n] << " " << structure.x[n] << " " << structure.y[n] << " "
           << structure.z[n] << " " << structure.fx[n] << " " << structure.fy[n] << " "
           << structure.fz[n] << "\n";
  }
}

static void write(
  const std::string& outputfile,
  const std::vector<Structure>& structures)
{
  std::ofstream output(outputfile);
  if (!output.is_open()) {
    std::cout << "Failed to open " << outputfile << std::endl;
    exit(1);
  }
  std::cout << outputfile << " is opened." << std::endl;
  for (int nc = 0; nc < structures.size(); ++nc) {
    write_one_structure(output, structures[nc]);
  }
  output.close();
  std::cout << outputfile << " is closed." << std::endl;
}

static void set_energy_weight_to_zero(std::vector<Structure>& structures)
{
  for (int nc = 0; nc < structures.size(); ++nc) {
    structures[nc].energy_weight = 0;
  }
}

static void set_box_to_1000(std::vector<Structure>& structures)
{
  for (int nc = 0; nc < structures.size(); ++nc) {
    structures[nc].box[0] = 1000;
    structures[nc].box[1] = 0;
    structures[nc].box[2] = 0;
    structures[nc].box[3] = 0;
    structures[nc].box[4] = 1000;
    structures[nc].box[5] = 0;
    structures[nc].box[6] = 0;
    structures[nc].box[7] = 0;
    structures[nc].box[8] = 1000;
  }
}

static void change_sid(std::vector<Structure>& structures, const std::string& new_sid)
{
  for (int nc = 0; nc < structures.size(); ++nc) {
    structures[nc].has_sid = true;
    structures[nc].sid = new_sid;
  }
}

static double get_volume(const double* box)
{
  return std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) +
         box[1] * (box[5] * box[6] - box[3] * box[8]) +
         box[2] * (box[3] * box[7] - box[4] * box[6]));
}

static std::vector<std::string> get_atom_symbols(const std::string& nep_file)
{
  std::ifstream input_potential(nep_file);
  if (!input_potential.is_open()) {
    std::cout << "Failed to open " << nep_file << std::endl;
    exit(1);
  }

  std::string potential_name;
  input_potential >> potential_name;
  int number_of_types;
  input_potential >> number_of_types;
  std::vector<std::string> atom_symbols(number_of_types);
  for (int n = 0; n < number_of_types; ++n) {
    input_potential >> atom_symbols[n];
  }

  input_potential.close();
  return atom_symbols;
}

static void calculate_one_structure(
  NEP3& nep3,
  std::vector<std::string>& atom_symbols,
  Structure& structure,
  const std::string& functional,
  double D3_cutoff,
  double D3_cutoff_cn)
{
  std::vector<double> box(9);
  for (int d1 = 0; d1 < 3; ++d1) {
    for (int d2 = 0; d2 < 3; ++d2) {
      box[d1 * 3 + d2] = structure.box[d2 * 3 + d1];
    }
  }

  std::vector<int> type(structure.num_atom);
  std::vector<double> position(structure.num_atom * 3);
  std::vector<double> potential(structure.num_atom);
  std::vector<double> force(structure.num_atom * 3);
  std::vector<double> virial(structure.num_atom * 9);

  for (int n = 0; n < structure.num_atom; n++) {
    position[n] = structure.x[n];
    position[n + structure.num_atom] = structure.y[n];
    position[n + structure.num_atom * 2] = structure.z[n];

    bool is_allowed_element = false;
    for (int t = 0; t < atom_symbols.size(); ++t) {
      if (structure.atom_symbol[n] == atom_symbols[t]) {
        type[n] = t;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      std::cout << "There is atom not allowed in the used NEP potential.\n";
      exit(1);
    }
  }

  nep3.compute_dftd3(functional, D3_cutoff, D3_cutoff_cn, type, box, position, potential, force, virial);

  for (int n = 0; n < structure.num_atom; n++) {
    structure.energy += potential[n];
    structure.fx[n] += force[0 * structure.num_atom + n];
    structure.fy[n] += force[1 * structure.num_atom + n];
    structure.fz[n] += force[2 * structure.num_atom + n];
  }
  if (structure.has_virial) {
    for (int d = 0; d < 9; ++d) {
      for (int n = 0; n < structure.num_atom; n++) {
        structure.virial[d] += virial[d * structure.num_atom + n];
      }
    }
  } else if (structure.has_stress) {
    for (int d = 0; d < 9; ++d) {
      for (int n = 0; n < structure.num_atom; n++) {
        structure.stress[d] -= virial[d * structure.num_atom + n] / get_volume(structure.box);
      }
    }
  }
}

static void add_d3(std::vector<Structure>& structures, const std::string& functional)
{
  NEP3 nep3("nep.txt");
  std::vector<std::string> atom_symbols = get_atom_symbols("nep.txt");
  for (int nc = 0; nc < structures.size(); ++nc) {
    calculate_one_structure(nep3, atom_symbols, structures[nc], functional, 12, 6);
  }
}

static void split_into_accurate_and_inaccurate(
  const std::vector<Structure>& structures, 
  double energy_threshold, 
  double force_threshold,
  double virial_threshold)
{
  std::ifstream input_energy("energy_train.out");
  std::ifstream input_force("force_train.out");
  std::ifstream input_virial("virial_train.out");
  std::ofstream output_accurate("accurate.xyz");
  std::ofstream output_inaccurate("inaccurate.xyz");
  int num1 = 0;
  int num2 = 0;
  for (int nc = 0; nc < structures.size(); ++nc) {
    bool force_is_small = true;
    for (int n = 0; n < structures[nc].num_atom; ++n) {
      double fx = structures[nc].fx[n];
      double fy = structures[nc].fy[n];
      double fz = structures[nc].fz[n];
      if (fx * fx + fy * fy + fz * fz > 1600.0) {
        force_is_small = false;
        break;
      }
    }

    bool is_accurate = true;

    double energy_nep = 0.0;
    double energy_ref = 0.0;
    input_energy >> energy_nep >> energy_ref;

    if (structures[nc].energy_weight > 0.5f && energy_threshold > 0) {
      if (std::abs(energy_nep - energy_ref) > energy_threshold) {
        is_accurate = false;
      }
    }

    double force_nep[3];
    double force_ref[3];
    for (int n = 0; n < structures[nc].num_atom; ++n) {
      input_force >> force_nep[0] >> force_nep[1] >> force_nep[2] >> force_ref[0] >> force_ref[1] >> force_ref[2];
      double fx_diff = force_nep[0] - force_ref[0];
      double fy_diff = force_nep[1] - force_ref[1];
      double fz_diff = force_nep[2] - force_ref[2];
      if (fx_diff * fx_diff + fy_diff * fy_diff + fz_diff * fz_diff > force_threshold * force_threshold) {
        is_accurate = false;
      }
    }

    double virial_nep[6];
    double virial_ref[6];
    for (int n = 0; n < 6; ++n) {
      input_virial >> virial_nep[n];
    }
    for (int n = 0; n < 6; ++n) {
      input_virial >> virial_ref[n];
    }
    for (int n = 0; n < 6; ++n) {
      if (std::abs(virial_nep[n] - virial_ref[n]) > virial_threshold) {
        if (structures[nc].has_virial || structures[nc].has_stress) {
          is_accurate = false;
        }
      }
    }

    //if (force_is_small) {
      if (is_accurate) {
        write_one_structure(output_accurate, structures[nc]);
        num1++;
      } else{
        write_one_structure(output_inaccurate, structures[nc]);
        num2++;
      }
    //}
  }
  input_energy.close();
  input_force.close();
  input_virial.close();
  output_accurate.close();
  output_inaccurate.close();
  std::cout << "Number of structures written into accurate.xyz = " << num1 << std::endl;
  std::cout << "Number of structures written into inaccurate.xyz = " << num2 << std::endl;
}

static void calculate_mae_and_rmse_one(
  const std::string& filename, 
  const std::string& units,
  const int num_components)
{
  const int num_tokens = num_components * 2;
  std::ifstream input(filename);
  int count = 0;
  std::vector<double> data(num_tokens);

  double mae = 0.0;
  double rmse = 0.0;

  if (!input.is_open()) {
    std::cout << "Failed to open " << filename << std::endl;
    exit(1);
  } else {
    while (true) {
      std::vector<std::string> tokens = get_tokens(input);
      if (tokens.size() == 0) {
        break;
      } else if (tokens.size() != num_tokens) {
        std::cout << "Number of values per row should be " << num_tokens << std::endl;
        exit(1);
      }
      for (int d = 0; d < num_tokens; ++d) {
        data[d] = get_double_from_token(tokens[d], __FILE__, __LINE__);
      }
      bool is_valid_data = true;
      for (int d = 0; d < num_components; ++d) {
        double diff = std::abs(data[d] - data[d + num_components]);
        if (num_components != 6 || data[d + num_components] > -1.0e3) {
          mae += diff;
          rmse += diff * diff; 
        } else {
          is_valid_data = false;
        }
      }
      if (is_valid_data) {
        count += num_components;
      }
    }
    input.close();
  }

  if (count > 0) {
    mae = mae / count;
    rmse = std::sqrt(rmse / count);
    std::cout << filename << "  MAE = " << mae << units << std::endl;
    std::cout << filename << " RMSE = " << rmse << units << std::endl;
  }
}

static void calculate_mae_and_rmse()
{
  calculate_mae_and_rmse_one("energy_train.out", " eV/atom", 1);
  calculate_mae_and_rmse_one("force_train.out", " eV/A", 3);
  calculate_mae_and_rmse_one("virial_train.out", " eV/atom", 6);
  calculate_mae_and_rmse_one("stress_train.out", " GPa", 6);
}

static void get_valid_structures(
  const std::vector<Structure>& structures, 
  double energy_threshold, 
  double force_threshold,
  double stress_threshold)
{
  std::ofstream output_valid("valid.xyz");
  std::ofstream output_invalid("invalid.xyz");
  int num1 = 0;
  int num2 = 0;
  for (int nc = 0; nc < structures.size(); ++nc) {
    bool is_valid = true;
    for (int n = 0; n < structures[nc].num_atom; ++n) {
      double fx = structures[nc].fx[n];
      double fy = structures[nc].fy[n];
      double fz = structures[nc].fz[n];
      if (fx * fx + fy * fy + fz * fz > force_threshold * force_threshold) {
        is_valid = false;
        break;
      }
    }
    if (structures[nc].energy_weight > 0.5f && energy_threshold > 0) {
      if (structures[nc].energy > energy_threshold) {
        is_valid = false;
      }
    }
    if (structures[nc].has_stress) {
      for (int n = 0; n < 9; ++n) {
        if (std::abs(structures[nc].stress[n] * 160.2) > stress_threshold) {
          is_valid = false;
          break;
        }
      }
    }
    if (structures[nc].has_virial) {
      for (int n = 0; n < 9; ++n) {
        if (std::abs(structures[nc].virial[n] / get_volume(structures[nc].box) * 160.2)  > stress_threshold) {
          is_valid = false;
          break;
        }
      }
    }

    if (is_valid) {
      write_one_structure(output_valid, structures[nc]);
      num1++;
    } else{
      write_one_structure(output_invalid, structures[nc]);
      num2++;
    }
  }
  output_valid.close();
  output_invalid.close();
  std::cout << "Number of structures written into valid.xyz = " << num1 << std::endl;
  std::cout << "Number of structures written into invalid.xyz = " << num2 << std::endl;
}

static void split_with_sid(const std::vector<Structure>& structures)
{
  std::ofstream output_ch("../ch/train.xyz");
  std::ofstream output_unep1("../unep1/train.xyz");
  std::ofstream output_hydrate("../hydrate/train.xyz");
  std::ofstream output_chonps("../chonps/train.xyz");
  std::ofstream output_spice("../spice/train.xyz");
  std::ofstream output_water("../water/train.xyz");
  std::ofstream output_mp("../mp/train.xyz");
  std::ofstream output_omat("../omat/train.xyz");
  std::ofstream output_protein("../protein/train.xyz");
  std::ofstream output_ani1xnr("../ani1xnr/train.xyz");
  std::ofstream output_sse_vasp("../sse_vasp/train.xyz");
  std::ofstream output_sse_abacus("../sse_abacus/train.xyz");
  std::ofstream output_cspbx("../cspbx/train.xyz");
  int num_ch = 0;
  int num_unep1 = 0;
  int num_hydrate = 0;
  int num_chonps = 0;
  int num_spice = 0;
  int num_omat = 0;
  int num_water = 0;
  int num_mp = 0;
  int num_protein = 0;
  int num_ani1xnr = 0;
  int num_sse_vasp = 0;
  int num_sse_abacus = 0;
  int num_cspbx = 0;
  for (int nc = 0; nc < structures.size(); ++nc) {
    if (structures[nc].sid == "ch") {
      write_one_structure(output_ch, structures[nc]);
        num_ch++;
    } else if (structures[nc].sid == "unep1") {
      write_one_structure(output_unep1, structures[nc]);
        num_unep1++;
    } else if (structures[nc].sid == "hydrate") {
      write_one_structure(output_hydrate, structures[nc]);
        num_hydrate++;
    } else if (structures[nc].sid == "chonps") {
      write_one_structure(output_chonps, structures[nc]);
        num_chonps++;
    } else if (structures[nc].sid == "spice") {
      write_one_structure(output_spice, structures[nc]);
        num_spice++;
    } else if (structures[nc].sid == "water") {
      write_one_structure(output_water, structures[nc]);
        num_water++;
    } else if (structures[nc].sid == "mp") {
      write_one_structure(output_mp, structures[nc]);
        num_mp++;
    } else if (structures[nc].sid == "protein") {
      write_one_structure(output_protein, structures[nc]);
        num_protein++;
    } else if (structures[nc].sid == "ani1xnr") {
      write_one_structure(output_ani1xnr, structures[nc]);
        num_ani1xnr++;
    } else if (structures[nc].sid == "sse_abacus") {
      write_one_structure(output_sse_abacus, structures[nc]);
        num_sse_abacus++;
    } else if (structures[nc].sid == "sse_vasp") {
      write_one_structure(output_sse_vasp, structures[nc]);
        num_sse_vasp++;
    } else if (structures[nc].sid == "omat") {
      write_one_structure(output_omat, structures[nc]);
        num_omat++;
    } else if (structures[nc].sid == "cspbx") {
      write_one_structure(output_cspbx, structures[nc]);
        num_cspbx++;
    }
  }
  output_ch.close();
  output_unep1.close();
  output_hydrate.close();
  output_chonps.close();
  output_spice.close();
  output_omat.close();
  output_water.close();
  output_mp.close();
  output_protein.close();
  output_ani1xnr.close();
  output_sse_abacus.close();
  output_sse_vasp.close();
  output_cspbx.close();
  std::cout << "Number of structures written into ch.xyz = " << num_ch << std::endl;
  std::cout << "Number of structures written into unep1.xyz = " << num_unep1 << std::endl;
  std::cout << "Number of structures written into hydrate.xyz = " << num_hydrate << std::endl;
  std::cout << "Number of structures written into chonps.xyz = " << num_chonps << std::endl;
  std::cout << "Number of structures written into spice.xyz = " << num_spice << std::endl;
  std::cout << "Number of structures written into water.xyz = " << num_water << std::endl;
  std::cout << "Number of structures written into mp.xyz = " << num_mp << std::endl;
  std::cout << "Number of structures written into omat.xyz = " << num_omat << std::endl;
  std::cout << "Number of structures written into protein.xyz = " << num_protein << std::endl;
  std::cout << "Number of structures written into ani1xnr.xyz = " << num_ani1xnr << std::endl;
  std::cout << "Number of structures written into sse_abacus.xyz = " << num_sse_abacus << std::endl;
  std::cout << "Number of structures written into sse_vasp.xyz = " << num_sse_vasp << std::endl;
  std::cout << "Number of structures written into cspbx.xyz = " << num_cspbx << std::endl;
}

static void fps(std::vector<Structure>& structures, double distance_square_min, int dim)
{
  std::ifstream input_descriptor("descriptor.out");
  std::ofstream output_selected("selected.xyz");
  std::ofstream output_not_selected("not_selected.xyz");
  std::ofstream output_index_selected("indices_selected.txt");
  std::ofstream output_index_not_selected("indices_not_selected.txt");
  std::vector<Structure> structures_selected;

  int num1 = 0;
  int num2 = 0;

  for (int nc = 0; nc < structures.size(); ++nc) {
    structures[nc].q.resize(dim);
    for (int d = 0; d < dim; ++d) {
      input_descriptor >> structures[nc].q[d];
    }
    if (nc == 0) {
      structures_selected.emplace_back(structures[nc]);
      output_index_selected << nc << "\n";
      num1++;
      write_one_structure(output_selected, structures[nc]);
    } else {
      bool to_be_selected = true;
      for (int m = 0; m < structures_selected.size(); ++m) {
        double distance_square = 0.0;
        for (int d = 0; d < dim; ++d) {
          double temp = (structures[nc].q[d] - structures_selected[m].q[d]);
          distance_square += temp * temp;
        }
        if (distance_square < distance_square_min) {
          to_be_selected = false;
          break;
        }
      }
      if (to_be_selected) {
        structures_selected.emplace_back(structures[nc]);
        output_index_selected << nc << "\n";
        num1++;
        if (num1 % 1000 == 0) {
          std::cout << "#selected = " << num1 << ", current structure ID = " << nc << "\n";
        }
        write_one_structure(output_selected, structures[nc]);
      } else {
        output_index_not_selected << nc << "\n";
        num2++;
        write_one_structure(output_not_selected, structures[nc]);
      }
    }
  }

  input_descriptor.close();
  output_selected.close();
  output_not_selected.close();
  output_index_selected.close();
  output_index_not_selected.close();
  std::cout << "Number of structures written into selected.xyz = " << num1 << std::endl;
  std::cout << "Number of structures written into not_selected.xyz = " << num2 << std::endl;
}

static void get_composition(std::vector<Structure>& structures)
{
  //int num_elements = 4;
  //std::string elements[] = {"H", "C", "N", "O"};

  //int num_elements = 10;
  //std::string elements[] = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"};

  int num_elements = 5;
  std::string elements[] = {"H", "C", "N", "O", "S"};

  std::ofstream output("count.txt");
  for (int nc = 0; nc < structures.size(); ++nc) {
    std::vector<int> counts(num_elements, 0);
    for (int n = 0; n < structures[nc].num_atom; ++n) {
      for (int i = 0; i < num_elements; ++i) {
        if (structures[nc].atom_symbol[n] == elements[i]) {
          ++counts[i];
          break;
        }
      }
    }
    for (int i = 0; i < num_elements; ++i) {
      output << counts[i] << " ";
    }
    output << "\n";
  }
  output.close();
}

static void shift_energy_multiple_species(std::vector<Structure>& structures)
{
  //int num_elements = 4;
  //std::string elements[] = {"H", "C", "N", "O"};
  //double delta_energy[] = {-3.2598,   -7.1457,   -7.6629,   -5.6500};

  /*int num_elements = 10;
  std::string elements[] = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"};
  double delta_energy[] = {
    0.001254026014518,   
    0.102890390836071,   
    0.148235407508952,   
    0.204097843036162,   
    0.271400629040059,   
    0.928501138100502,
    1.083209903212600,   
    1.252197567336926,   
    7.004441200065508,   
    0.810198990224952};
  for (int i = 0; i < num_elements; ++i) {
    delta_energy[i] *= 1.0e4;
  }*/

  int num_elements = 5;
  std::string elements[] = {"H", "C", "N", "O", "S"};
  double delta_energy[] = {-1.265276327295640,  -1.880752053284493,  -4.451413902725383,  -2.004796016966084,  -2.060841966465145};

  for (int nc = 0; nc < structures.size(); ++nc) {
    std::vector<int> counts(num_elements, 0);
    for (int n = 0; n < structures[nc].num_atom; ++n) {
      for (int i = 0; i < num_elements; ++i) {
        if (structures[nc].atom_symbol[n] == elements[i]) {
          ++counts[i];
          break;
        }
      }
    }
    for (int i = 0; i < num_elements; ++i) {
      structures[nc].energy += counts[i] * delta_energy[i];
    }
  }
}

static void get_structures_with_given_species(
  std::string& outputfile,
  std::vector<Structure>& structures,
  int num_species,
  std::vector<std::string>& given_species)
{
  std::ofstream output(outputfile);
  if (!output.is_open()) {
    std::cout << "Failed to open " << outputfile << std::endl;
    exit(1);
  }
  std::cout << outputfile << " is opened." << std::endl;

  for (int nc = 0; nc < structures.size(); ++nc) {

    int is_valid_structure = true;
    
    for (int n = 0; n < structures[nc].num_atom; ++ n) {

      bool match = false;
      for (int k = 0; k < num_species; ++k) {
        if (structures[nc].atom_symbol[n] == given_species[k]) {
          match = true;
          break;
        }
      }

      if (!match) {
        is_valid_structure = false;
        break;
      }
    }

    if (is_valid_structure) {
      write_one_structure(output, structures[nc]);
    }
  }

  output.close();
  std::cout << outputfile << " is closed." << std::endl;
}

static void get_structures_with_given_species_no_subsystems(
  std::string& outputfile,
  std::vector<Structure>& structures,
  int num_species,
  std::vector<std::string>& given_species)
{
  std::ofstream output(outputfile);
  if (!output.is_open()) {
    std::cout << "Failed to open " << outputfile << std::endl;
    exit(1);
  }
  std::cout << outputfile << " is opened." << std::endl;

  for (int nc = 0; nc < structures.size(); ++nc) {

    std::vector<bool> has_at_least_one(num_species);
    for (int k = 0; k < num_species; ++k) {
      has_at_least_one[k] = false;
    }

    int is_valid_structure = true;
    
    for (int n = 0; n < structures[nc].num_atom; ++ n) {

      bool match = false;
      for (int k = 0; k < num_species; ++k) {
        if (structures[nc].atom_symbol[n] == given_species[k]) {
          match = true;
          has_at_least_one[k] = true;
          //break;
        }
      }

      if (!match) {
        is_valid_structure = false;
        break;
      }
    }

    for (int k = 0; k < num_species; ++k) {
      is_valid_structure = is_valid_structure && has_at_least_one[k];
    }

    if (is_valid_structure) {
      write_one_structure(output, structures[nc]);
    }
  }

  output.close();
  std::cout << outputfile << " is closed." << std::endl;
}

int main(int argc, char* argv[])
{
  std::cout << "====================================================\n";
  std::cout << "Welcome to use nep_data_toolkit!" << std::endl;
  std::cout << "Here are the functionalities:" << std::endl;
  std::cout << "----------------------------------------------------\n";
  std::cout << "1: count the number of structures\n";
  std::cout << "2: copy\n";
  std::cout << "3: split into accurate.xyz and inaccurate.xyz\n";
  std::cout << "4: split according to sid\n";
  std::cout << "5: descriptor-space subsampling\n";
  std::cout << "6: set energy_weight to zero\n";
  std::cout << "7: add or change sid\n";
  std::cout << "8: add D3\n";
  std::cout << "9: get composition\n";
  std::cout << "10: shift energy for multiple species\n";
  std::cout << "11: get structures with given species\n";
  std::cout << "12: change box to 1000\n";
  std::cout << "13: get valid structures\n";
  std::cout << "14: calculate MAEs and RMSEs\n";
  std::cout << "====================================================\n";

  std::cout << "Please choose a number based on your purpose: ";
  int option;
  std::cin >> option;

  if (option == 1) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
  } else if (option == 2) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    write(output_filename, structures_input);
  } else if (option == 3) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the energy threshold in units of eV/atom (negative to ignore): ";
    double energy_threshold;
    std::cin >> energy_threshold;
    std::cout << "Please enter the force threshold in units of eV/A: ";
    double force_threshold;
    std::cin >> force_threshold;
    std::cout << "Please enter the virial threshold in units of eV/atom: ";
    double virial_threshold;
    std::cin >> virial_threshold;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    split_into_accurate_and_inaccurate(structures_input, energy_threshold, force_threshold, virial_threshold);
  } else if (option == 4) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    split_with_sid(structures_input);
  } else if (option == 5) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the minimal distance in descriptor space: ";
    double distance;
    std::cin >> distance;
    std::cout << "Please enter the dimension of descriptor space: ";
    int dim;
    std::cin >> dim;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;

    clock_t time_begin = clock();
    fps(structures_input, distance * distance, dim);
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
    std::cout << "Time used for descriptor-space subsampling = " << time_used << " s.\n";
  } else if (option == 6) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    set_energy_weight_to_zero(structures_input);
    write(output_filename, structures_input);
  } else if (option == 7) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::cout << "Please enter the sid to be used for all the structures: ";
    std::string sid;
    std::cin >> sid;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    change_sid(structures_input, sid);
    write(output_filename, structures_input);
  } else if (option == 8) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::cout << "Please enter the DFT functional: ";
    std::string functional;
    std::cin >> functional;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    add_d3(structures_input, functional);
    write(output_filename, structures_input);
  } else if (option == 9) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    get_composition(structures_input);
  } else if (option == 10) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    shift_energy_multiple_species(structures_input);
    write(output_filename, structures_input);
  } else if (option == 11) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::cout << "Please enter the number of species: ";
    int num_species;
    std::cin >> num_species;
    std::vector<std::string> given_species(num_species);
    for (int n = 0; n < num_species; ++n) {
      std::cout << "Please enter species " << n << ": ";
      std::cin >> given_species[n];
    }
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    get_structures_with_given_species(output_filename, structures_input, num_species, given_species);
  } else if (option == 12) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the output xyz filename: ";
    std::string output_filename;
    std::cin >> output_filename;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    set_box_to_1000(structures_input);
    write(output_filename, structures_input);
  } else if (option == 13) {
    std::cout << "Please enter the input xyz filename: ";
    std::string input_filename;
    std::cin >> input_filename;
    std::cout << "Please enter the energy threshold in units of eV/atom (negative to ignore): ";
    double energy_threshold;
    std::cin >> energy_threshold;
    std::cout << "Please enter the force threshold in units of eV/A: ";
    double force_threshold;
    std::cin >> force_threshold;
    std::cout << "Please enter the stress threshold in units of GPa: ";
    double stress_threshold;
    std::cin >> stress_threshold;
    std::vector<Structure> structures_input;
    read(input_filename, structures_input);
    std::cout << "Number of structures read from "
              << input_filename + " = " << structures_input.size() << std::endl;
    get_valid_structures(structures_input, energy_threshold, force_threshold, stress_threshold);
  } else if (option == 14) {
    calculate_mae_and_rmse();
  } else {
    std::cout << "This is an invalid option.";
    exit(1);
  }

  std::cout << "Done." << std::endl;
  return EXIT_SUCCESS;
}


