/*-----------------------------------------------------------------------------------------------100
compile:
    g++ -O3 select_xyz.cpp
run:
    ./a.out 0 max_atom_0 input_dir output_dir
    # read in input_dir/train.xyz and put the structures with the number of atoms smaller than
    # num_atom_0 to output_dir/train.xyz and the others to output_dir/test.xyz


    ./a.out 1 energy_error_0 input_dir output_dir
    # first copy input_dir/train.xyz to output_dir/train.xyz and then read in input_dir/test.xyz and
    # input_dir/energy_test.out and put the structures with energy error larger than energy_error_0
    # (in units of eV/atom) to output_dir/train.xyz (append) and the others to output_dir/test.xyz
--------------------------------------------------------------------------------------------------*/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

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

float get_float_from_token(const std::string& token, const char* filename, const int line)
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
  int has_virial;
  float energy;
  float weight;
  float virial[9];
  float box[9];
  std::vector<std::string> atom_symbol;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
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
    structure.x[na] = get_float_from_token(tokens[0 + pos_offset], __FILE__, __LINE__);
    structure.y[na] = get_float_from_token(tokens[1 + pos_offset], __FILE__, __LINE__);
    structure.z[na] = get_float_from_token(tokens[2 + pos_offset], __FILE__, __LINE__);
    if (num_columns > 4) {
      structure.fx[na] = get_float_from_token(tokens[0 + force_offset], __FILE__, __LINE__);
      structure.fy[na] = get_float_from_token(tokens[1 + force_offset], __FILE__, __LINE__);
      structure.fz[na] = get_float_from_token(tokens[2 + force_offset], __FILE__, __LINE__);
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

  bool has_energy_in_exyz = false;
  for (const auto& token : tokens) {
    const std::string energy_string = "energy=";
    if (token.substr(0, energy_string.length()) == energy_string) {
      has_energy_in_exyz = true;
      structure.energy = get_float_from_token(
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
      structure.weight = get_float_from_token(
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
        structure.box[m] = get_float_from_token(
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
        structure.virial[m] = get_float_from_token(
          tokens[n + m].substr(
            (m == 0) ? (virial_string.length() + 1) : 0,
            (m == 8) ? (tokens[n + m].length() - 1) : tokens[n + m].length()),
          __FILE__, __LINE__);
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
  output << "lattice=\"";
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

  output << "Properties=species:S:1:pos:R:3:force:R:3\n";

  for (int n = 0; n < structure.num_atom; ++n) {
    output << structure.atom_symbol[n] << " " << structure.x[n] << " " << structure.y[n] << " "
           << structure.z[n] << " " << structure.fx[n] << " " << structure.fy[n] << " "
           << structure.fz[n] << "\n";
  }
}

const std::string ELEMENTS[89] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",
  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
  "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re",
  "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

const int INDEX_START[16] = {0,5,10,15,20,25,30,35,41,47,53,59,65,71,77,83};
const int INDEX_END[16] = {5,10,15,20,25,30,35,41,47,53,59,65,71,77,83,89};

std::vector<std::string> get_elements(const Structure& structure) 
{
  std::vector<std::string> elements;
  for (int n = 0; n < structure.num_atom; ++n) {
    bool has_same_element = false;
    for (int i = 0; i < elements.size(); ++i) {
      if (structure.atom_symbol[n] == elements[i]) {
        has_same_element = true;
        break;
      }
    }
    if (!has_same_element) {
      elements.emplace_back(structure.atom_symbol[n]);
    }
  }
  return elements;
}

bool elements_is_in_group(const std::vector<std::string>& elements, const int index_start, const int index_end) 
{
  bool answer = true;
  for (int i = 0; i < elements.size(); ++i) {
    bool has_same_element = false;
    for (int j = index_start; j < index_end; ++j) {
      if (elements[i] == ELEMENTS[j]) {
        has_same_element = true;
        break;
      }
    }
    if (!has_same_element) {
      answer = false;
      break;
    }
  }
  return answer;
}

int main(int argc, char* argv[])
{

  clock_t time_begin = clock();


  std::vector<Structure> structures;
  read("train.xyz", structures);
  std::cout << "Number of structures read in = " << structures.size() << std::endl;

  std::vector<std::ofstream> outputs16(16);
  for (int g = 0; g < 16; ++g) {
    outputs16[g].open("output16/" + std::to_string(g) + ".xyz", std::ios::out);
  }
  std::vector<std::ofstream> outputs8(8);
  for (int g = 0; g < 8; ++g) {
    outputs8[g].open("output8/" + std::to_string(g) + ".xyz", std::ios::out);
  }
  std::vector<std::ofstream> outputs4(4);
  for (int g = 0; g < 4; ++g) {
    outputs4[g].open("output4/" + std::to_string(g) + ".xyz", std::ios::out);
  }
  std::vector<std::ofstream> outputs2(2);
  for (int g = 0; g < 2; ++g) {
    outputs2[g].open("output2/" + std::to_string(g) + ".xyz", std::ios::out);
  }
  std::vector<std::ofstream> outputs1(1);
  for (int g = 0; g < 1; ++g) {
    outputs1[g].open("output1/" + std::to_string(g) + ".xyz", std::ios::out);
  }

  int count16=0;
  int count8=0;
  int count4=0;
  int count2=0;
  int count1=0;
  for (int nc = 0; nc < structures.size(); ++ nc) {
    bool is_written = false;

    // 16
    for (int g = 0; g < 16; ++g) {
      if (elements_is_in_group(get_elements(structures[nc]), INDEX_START[g], INDEX_END[g])) {
        write_one_structure(outputs16[g], structures[nc]);
        is_written = true;
        count16++;
        //std::cout << "g16_selcted=" << g  << std::endl;
        break;
      }
    }

    if (is_written) continue;

    // 8
    for (int g = 0; g < 8; ++g) {
      if (elements_is_in_group(get_elements(structures[nc]), INDEX_START[g*2], INDEX_END[g*2+1])) {
        write_one_structure(outputs8[g], structures[nc]);
        is_written = true;
        count8++;
        //std::cout << "g8_selcted=" << g  << std::endl;
        break;
      }
    }
    
    if (is_written) continue;

    // 4
    for (int g = 0; g < 4; ++g) {
      if (elements_is_in_group(get_elements(structures[nc]), INDEX_START[g*4], INDEX_END[g*4+3])) {
        write_one_structure(outputs4[g], structures[nc]);
        is_written = true;
        count4++;
        //std::cout << "g4_selcted=" << g  << std::endl;
        break;
      }
    }
    
    if (is_written) continue;

    // 2
    for (int g = 0; g < 2; ++g) {
      if (elements_is_in_group(get_elements(structures[nc]), INDEX_START[g*8], INDEX_END[g*8+7])) {
        write_one_structure(outputs2[g], structures[nc]);
        is_written = true;
        count2++;
        //std::cout << "g2_selcted=" << g  << std::endl;
        break;
      }
    }
    
    if (is_written) continue;

    // 1
    for (int g = 0; g < 1; ++g) {
      if (elements_is_in_group(get_elements(structures[nc]), INDEX_START[g*16], INDEX_END[g*16+15])) {
        write_one_structure(outputs1[g], structures[nc]);
        is_written = true;
        count1++;
        //std::cout << "g1_selcted=" << g  << std::endl;
        break;
      }
    }
    
    if (is_written) continue;

  }

  for (int g = 0; g < 16; ++g) {
    outputs16[g].close();
  }
  for (int g = 0; g < 8; ++g) {
    outputs8[g].close();
  }
  for (int g = 0; g < 4; ++g) {
    outputs4[g].close();
  }
  for (int g = 0; g < 2; ++g) {
    outputs2[g].close();
  }
  for (int g = 0; g < 1; ++g) {
    outputs1[g].close();
  }

  std::cout << "count16=" << count16 << std::endl;
  std::cout << "count8=" << count8 << std::endl;
  std::cout << "count4=" << count4 << std::endl;
  std::cout << "count2=" << count2 << std::endl;
  std::cout << "count1=" << count1 << std::endl;
  std::cout << "Done." << std::endl;

  clock_t time_finish = clock();
  double time_used = (time_finish - time_begin) / double(CLOCKS_PER_SEC);
  std::cout << "time used = " << time_used << " seconds." << std::endl;

  return EXIT_SUCCESS;
}
