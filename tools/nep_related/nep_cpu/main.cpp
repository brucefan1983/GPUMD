#include "nep.h"
#include <fstream>
#include <iostream>
#include <time.h>

struct Atom {
  int N;
  Box box;
  std::vector<int> type;
  std::vector<double> position, potential, force, virial;
};

static std::vector<std::string> get_atom_symbols()
{
  std::ifstream input_potential("nep.txt");
  if (!input_potential.is_open()) {
    std::cout << "Error: cannot open nep.txt.\n";
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

static void readXYZ(Atom& atom)
{
  std::cout << "Reading xyz.in.\n";

  std::ifstream input_file("xyz.in");

  if (!input_file) {
    std::cout << "Cannot open xyz.in\n";
    exit(1);
  }

  input_file >> atom.N;
  std::cout << "    Number of atoms is " << atom.N << ".\n";

  input_file >> atom.box.cpu_h[0];
  input_file >> atom.box.cpu_h[3];
  input_file >> atom.box.cpu_h[6];
  input_file >> atom.box.cpu_h[1];
  input_file >> atom.box.cpu_h[4];
  input_file >> atom.box.cpu_h[7];
  input_file >> atom.box.cpu_h[2];
  input_file >> atom.box.cpu_h[5];
  input_file >> atom.box.cpu_h[8];
  atom.box.get_inverse();

  std::cout << "    Box matrix h = [a, b, c] is\n";
  for (int d1 = 0; d1 < 3; ++d1) {
    for (int d2 = 0; d2 < 3; ++d2) {
      std::cout << "\t" << atom.box.cpu_h[d1 * 3 + d2];
    }
    std::cout << "\n";
  }

  std::cout << "    Inverse box matrix g = inv(h) is\n";
  for (int d1 = 0; d1 < 3; ++d1) {
    for (int d2 = 0; d2 < 3; ++d2) {
      std::cout << "\t" << atom.box.cpu_h[9 + d1 * 3 + d2];
    }
    std::cout << "\n";
  }

  std::vector<std::string> atom_symbols = get_atom_symbols();

  atom.type.resize(atom.N);
  atom.position.resize(atom.N * 3);
  atom.potential.resize(atom.N);
  atom.force.resize(atom.N * 3);
  atom.virial.resize(atom.N * 9);

  for (int n = 0; n < atom.N; n++) {
    std::string atom_symbol_tmp;
    input_file >> atom_symbol_tmp >> atom.position[n] >> atom.position[n + atom.N] >>
      atom.position[n + atom.N * 2];
    bool is_allowed_element = false;
    for (int t = 0; t < atom_symbols.size(); ++t) {
      if (atom_symbol_tmp == atom_symbols[t]) {
        atom.type[n] = t;
        is_allowed_element = true;
      }
    }
    if (!is_allowed_element) {
      std::cout << "There is atom in xyz.in that is not allowed in the used NEP potential.\n";
      exit(1);
    }
  }
}

int main(int argc, char* argv[])
{
  Atom atom;
  readXYZ(atom);
  NEP3 nep3(atom.N, atom.N);

  clock_t time_begin = clock();

  for (int n = 0; n < 100; ++n) {
    nep3.compute(atom.box, atom.type, atom.position, atom.potential, atom.force, atom.virial);
  }

  clock_t time_finish = clock();
  float time_used = (time_finish - time_begin) / float(CLOCKS_PER_SEC);
  std::cout << "Time used for NEP calculations = " << time_used << " s.\n";

  float speed = atom.N * 100 / time_used;
  float cost = 1000 / speed;
  std::cout << "Computational speed = " << speed << " atom-step/second.\n";
  std::cout << "Computational cost = " << cost << " microsecond/atom-step.\n";

  std::ofstream output_file("force_cpu.out");

  if (!output_file.is_open()) {
    std::cout << "Cannot open force_cpu.out\n";
    exit(1);
  }
  for (int n = 0; n < atom.N; ++n) {
    output_file << atom.force[n] << " " << atom.force[n + atom.N] << " "
                << atom.force[n + atom.N * 2] << "\n";
  }
  output_file.close();

  return 0;
}
