#include "nep.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <time.h>

const int MN = 1000;

struct ExpandedBox {
  int num_cells[3];
  float h[18];
};

class Box
{
public:
  double cpu_h[18];                                   // the box data
  double thickness_x = 0.0;                           // thickness perpendicular to (b x c)
  double thickness_y = 0.0;                           // thickness perpendicular to (c x a)
  double thickness_z = 0.0;                           // thickness perpendicular to (a x b)
  double get_area(const int d) const;                 // get the area of one face
  double get_volume(void) const;                      // get the volume of the box
  void get_inverse(void);                             // get the inverse box matrix
  bool get_num_bins(const double rc, int num_bins[]); // get the number of bins in each direction
};

static float get_area_one_direction(const double* a, const double* b)
{
  double s1 = a[1] * b[2] - a[2] * b[1];
  double s2 = a[2] * b[0] - a[0] * b[2];
  double s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

double Box::get_area(const int d) const
{
  double area;
  double a[3] = {cpu_h[0], cpu_h[3], cpu_h[6]};
  double b[3] = {cpu_h[1], cpu_h[4], cpu_h[7]};
  double c[3] = {cpu_h[2], cpu_h[5], cpu_h[8]};
  if (d == 0) {
    area = get_area_one_direction(b, c);
  } else if (d == 1) {
    area = get_area_one_direction(c, a);
  } else {
    area = get_area_one_direction(a, b);
  }
  return area;
}

double Box::get_volume(void) const
{
  double volume = abs(
    cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
    cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
    cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]));
  return volume;
}

void Box::get_inverse(void)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  double det = cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
               cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
               cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= det;
  }
}

static void get_expanded_box(const double rc, const Box& box, ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = int(ceil(2.0 * rc / thickness_x));
  ebox.num_cells[1] = int(ceil(2.0 * rc / thickness_y));
  ebox.num_cells[2] = int(ceil(2.0 * rc / thickness_z));

  ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
  ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
  ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
  ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
  ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
  ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
  ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
  ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
  ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

  ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
  ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
  ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
  ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
  ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
  ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
  ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
  ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
  ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
  double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
               ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
               ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
  for (int n = 9; n < 18; n++) {
    ebox.h[n] /= det;
  }
}

struct Atom {
  int N;
  Box box;
  ExpandedBox ebox;
  std::vector<int> type;
  std::vector<double> position, potential, force, virial;
  std::vector<int> NN_radial;
  std::vector<int> NL_radial;
  std::vector<int> NN_angular;
  std::vector<int> NL_angular;
  std::vector<float> r12;
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
  atom.NN_radial.resize(atom.N);
  atom.NL_radial.resize(atom.N * MN);
  atom.NN_angular.resize(atom.N);
  atom.NL_angular.resize(atom.N * MN);
  atom.r12.resize(atom.N * MN * 6);
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

static void
apply_mic_small_box(const Box& box, const ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
  double sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
  double sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
  double sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
  sx12 -= nearbyint(sx12);
  sy12 -= nearbyint(sy12);
  sz12 -= nearbyint(sz12);
  x12 = ebox.h[0] * sx12 + ebox.h[1] * sy12 + ebox.h[2] * sz12;
  y12 = ebox.h[3] * sx12 + ebox.h[4] * sy12 + ebox.h[5] * sz12;
  z12 = ebox.h[6] * sx12 + ebox.h[7] * sy12 + ebox.h[8] * sz12;
}

static void find_neighbor_list_small_box(
  const float rc_radial,
  const float rc_angular,
  const int N,
  const Box box,
  const ExpandedBox ebox,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  for (int n1 = 0; n1 < N; ++n1) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = 0; n2 < N; ++n2) {
      for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
        for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
          for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }

            double delta[3];
            delta[0] = box.cpu_h[0] * ia + box.cpu_h[1] * ib + box.cpu_h[2] * ic;
            delta[1] = box.cpu_h[3] * ia + box.cpu_h[4] * ib + box.cpu_h[5] * ic;
            delta[2] = box.cpu_h[6] * ia + box.cpu_h[7] * ib + box.cpu_h[8] * ic;

            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(box, ebox, x12, y12, z12);

            float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
            if (distance_square < rc_radial * rc_radial) {
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = float(x12);
              g_y12_radial[count_radial * N + n1] = float(y12);
              g_z12_radial[count_radial * N + n1] = float(z12);
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = float(x12);
              g_y12_angular[count_angular * N + n1] = float(y12);
              g_z12_angular[count_angular * N + n1] = float(z12);
              count_angular++;
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

int main(int argc, char* argv[])
{
  Atom atom;
  readXYZ(atom);
  NEP3 nep3(atom.N);

  get_expanded_box(nep3.paramb.rc_radial, atom.box, atom.ebox);

  const int size_x12 = atom.NL_radial.size();

  find_neighbor_list_small_box(
    nep3.paramb.rc_radial, nep3.paramb.rc_angular, atom.N, atom.box, atom.ebox,
    atom.position.data(), atom.position.data() + atom.N, atom.position.data() + atom.N * 2,
    atom.NN_radial.data(), atom.NL_radial.data(), atom.NN_angular.data(), atom.NL_angular.data(),
    atom.r12.data(), atom.r12.data() + size_x12, atom.r12.data() + size_x12 * 2,
    atom.r12.data() + size_x12 * 3, atom.r12.data() + size_x12 * 4, atom.r12.data() + size_x12 * 5);

  clock_t time_begin = clock();

  for (int n = 0; n < 100; ++n) {
    nep3.compute(
      atom.NN_radial, atom.NL_radial, atom.NN_angular, atom.NL_angular, atom.type, atom.r12,
      atom.potential, atom.force, atom.virial);
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
