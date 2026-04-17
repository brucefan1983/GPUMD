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
Use finite difference to calculate the hessian (force constants).
    H_ij^ab = [F_i^a(-) - F_i^a(+)] / [u_j^b(+) - u_j^b(-)]
Then calculate the dynamical matrices with different k points.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "force/force_constant.cuh"
#include "hessian.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <vector>
#include <array>

namespace
{
  using Vec3 = std::array<double, 3>;
  using Mat3 = std::array<std::array<double, 3>, 3>;

  Vec3 matvec(const Mat3& m, const Vec3& v)
  {
    return Vec3{{
      m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
      m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
      m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]}};
  }

  Vec3 lerp(const Vec3& a, const Vec3& b, double t)
  {
    return Vec3{{a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])}};
  }

  Mat3 build_reciprocal_lattice(const Box& box, const int cxyz[3])
  {
    Mat3 rec;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        rec[i][j] = 2.0 * PI * cxyz[i] * box.cpu_h[9 + i * 3 + j];
      }
    }
    return rec;
  }
}

void Hessian::compute(
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group)
{
  initialize(atom.cpu_mass, box, force, atom.number_of_atoms);
  find_H(force, box, atom, group);

  if (num_kpoints == 1) // currently for Alex's GKMA calculations
  {
    find_D(box, atom);
    find_eigenvectors();
  } else {
    find_dispersion(box, atom);
  }
}

void Hessian::get_cutoff_from_potential(Force& force)
{
  for (const auto& potential : force.potentials) {
    cutoff = std::max(cutoff, potential->rc);
  }
  phonon_cutoff = cutoff * 2.0;
  printf("Using cutoff for phonon calculations: %g A.\n", phonon_cutoff);
}

void Hessian::create_basis(const std::vector<double>& cpu_mass, int N)
{
  num_basis = N / (cxyz[0] * cxyz[1] * cxyz[2]);

  basis.resize(num_basis);
  mass.resize(num_basis);
  for (int i = 0; i < num_basis; ++i) {
    basis[i] = i;
    mass[i] = cpu_mass[i];
  }

  label.resize(N);
  for (int n = 0; n < N; ++n) {
    int atom_idx = n % num_basis;
    label[n] = atom_idx;
  }
}

void Hessian::create_kpoints(const Box& box)
{
  std::ifstream input_kpoints("kpoints.in");
  if (!input_kpoints.is_open())
    PRINT_INPUT_ERROR("Cannot open kpoints.in file.");

  std::vector<std::vector<Vec3>> hsps;
  std::vector<Vec3> hsp;
  std::string line;
  std::string k_name;
  std::string k_names;

  while (std::getline(input_kpoints, line)) {
    auto tokens = get_tokens(line);
    if (tokens.empty()) {
      if (!hsp.empty()) {
        hsps.push_back(hsp);
        hsp.clear();
        hsp_names.push_back(k_names);
        k_names.clear();
      }
      continue;
    }
    if (tokens[0][0] == '#') 
      continue;
    
    if (tokens.size() < 4) {
      PRINT_INPUT_ERROR("kpoints.in file at least 4 parameters for each line.");
    } else {
      double k[3];
      for (int i = 0; i < 3; ++i) {
        k[i] = get_double_from_token(tokens[i], __FILE__, __LINE__);
      }
      hsp.push_back(Vec3{{k[0], k[1], k[2]}});
      k_name = tokens[3];
      if (!k_names.empty())
        k_names += " ";
      k_names += k_name;
    }
  }
  if (!hsp.empty()) {
    hsps.push_back(hsp);
    hsp_names.push_back(k_names);
  }

  num_kpoints = 1 - hsps.size();
  for (const auto& hsp : hsps)
    num_kpoints += hsp.size();
  num_kpoints = (num_kpoints - 1) * 100 + 1;
  kpoints.resize(num_kpoints * 3);
  kpath.resize(num_kpoints);

  double kpath_len = 0.0;
  const Mat3 rec_lat = build_reciprocal_lattice(box, cxyz);
  Vec3 k_first = matvec(rec_lat, hsps[0][0]);
  for (int i = 0; i < 3; ++i) {
    kpoints[i] = k_first[i];
  }
  kpath[0] = kpath_len;
  kpath_sym.push_back(kpath_len);

  int k_idx = 1;
  for (const auto& hsp : hsps) {
    for (int i = 1; i < hsp.size(); ++i) {
      const auto& start = matvec(rec_lat, hsp[i - 1]);
      const auto& end = matvec(rec_lat, hsp[i]);
      auto last = start;

      for (int j = 1; j <= 100; ++j) {
        auto kpt = lerp(start, end, j * 0.01);
        for (int i = 0; i < 3; ++i) {
          kpoints[k_idx * 3 + i] = kpt[i];
        }
        
        double d[3];
        for (int i = 0; i < 3; ++i) {
          d[i] = kpt[i] - last[i];
        }
        kpath_len += std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        kpath[k_idx] = kpath_len;
        last = kpt;

        if (j == 100)
          kpath_sym.push_back(kpath_len);
        ++k_idx;
      }
    }
  }
}

void Hessian::initialize(
  const std::vector<double>& cpu_mass, Box& box, Force& force, int N)
{
  get_cutoff_from_potential(force);

  std::ifstream fin("run.in");
  std::string line;
  bool has_rep = false;
  while (std::getline(fin, line)) {
    auto tokens = get_tokens(line);
    if (!tokens.empty() && tokens[0][0] != '#' && tokens[0] == "replicate") {  // 跳过空行和注释行
      has_rep = true;
      for (int i = 0; i < 3; ++i) {
        cxyz[i] = get_int_from_token(tokens[i + 1], __FILE__, __LINE__);
      }
    }
    break;
  }
  fin.close();
  if (!has_rep) {
    PRINT_INPUT_ERROR("replicate keyword not found in run.in file.");
  }

  int s_c[3] = {1, 1, 1};
  int stru_pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  double volume = box.get_volume();
  for (int i= 0; i < 3; ++i){
    double thickness = volume / box.get_area(i);
    double ori_thick = thickness / cxyz[i];
    if (stru_pbc[i]) {
      for (int j= 1;j< 100;++j){
        if (ori_thick *j>= cutoff *4){
          s_c[i] = j;
          break;
        }
      }
    }
  }
  printf("Suggested replicate size for phonon calculations:\n");
  printf("Replicate in x >= %d.\n", s_c[0]);
  printf("Replicate in y >= %d.\n", s_c[1]);
  printf("Replicate in z >= %d.\n", s_c[2]);

  create_basis(cpu_mass, N);
  create_kpoints(box);
  size_t num_H = num_basis * N * 9;
  size_t num_D = num_basis * num_basis * 9 * num_kpoints;
  H.resize(num_H, 0.0);
  DR.resize(num_D, 0.0);
  if (num_kpoints > 1) // for dispersion calculation
  {
    DI.resize(num_D, 0.0);
  }
}

bool Hessian::is_too_far(
  const Box& box,
  const std::vector<double>& cpu_position_per_atom,
  const size_t n1,
  const size_t n2)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;
  double x12 = cpu_position_per_atom[n2] - cpu_position_per_atom[n1];
  double y12 =
    cpu_position_per_atom[n2 + number_of_atoms] - cpu_position_per_atom[n1 + number_of_atoms];
  double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2] -
               cpu_position_per_atom[n1 + number_of_atoms * 2];
  apply_mic(box, x12, y12, z12);
  double d12_square = x12 * x12 + y12 * y12 + z12 * z12;
  return (d12_square > (phonon_cutoff * phonon_cutoff));
}

void Hessian::find_H(
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group)
{
  const int number_of_atoms = atom.number_of_atoms;

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, atom.cpu_position_per_atom, n1, n2)) {
        continue;
      }
      size_t offset = (nb * number_of_atoms + n2) * 9;
      find_H12(
        displacement,
        n1,
        n2,
        box,
        atom.position_per_atom,
        atom.type,
        group,
        atom.potential_per_atom,
        atom.force_per_atom,
        atom.virial_per_atom,
        force,
        H.data() + offset);
    }
  }
}

static void find_exp_ikr(
  const size_t n1,
  const size_t n2,
  const double* k,
  const Box& box,
  const std::vector<double>& cpu_position_per_atom,
  double& cos_kr,
  double& sin_kr)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;
  double x12 = cpu_position_per_atom[n2] - cpu_position_per_atom[n1];
  double y12 =
    cpu_position_per_atom[n2 + number_of_atoms] - cpu_position_per_atom[n1 + number_of_atoms];
  double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2] -
               cpu_position_per_atom[n1 + number_of_atoms * 2];
  apply_mic(box, x12, y12, z12);
  double kr = k[0] * x12 + k[1] * y12 + k[2] * z12;
  cos_kr = cos(kr);
  sin_kr = sin(kr);
}

void Hessian::output_D()
{
  FILE* fid = fopen("D.out", "w");
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t n1 = 0; n1 < num_basis * 3; ++n1) {
      for (size_t n2 = 0; n2 < num_basis * 3; ++n2) {
        // cuSOLVER requires column-major
        fprintf(fid, "%g ", DR[offset + n1 + n2 * num_basis * 3]);
      }
      if (num_kpoints > 1) {
        for (size_t n2 = 0; n2 < num_basis * 3; ++n2) {
          // cuSOLVER requires column-major
          fprintf(fid, "%g ", DI[offset + n1 + n2 * num_basis * 3]);
        }
      }
      fprintf(fid, "\n");
    }
  }
  fclose(fid);
}

void Hessian::find_omega(FILE* fid, size_t offset, size_t nk)
{
  size_t dim = num_basis * 3;
  std::vector<double> W(dim);
  eig_hermitian_QR(dim, DR.data() + offset, DI.data() + offset, W.data());
  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  fprintf(fid, "%.6f ", kpath[nk]);
  for (size_t n = 0; n < dim; ++n) {
    fprintf(fid, "%g ", W[n] * natural_to_THz);
  }
  fprintf(fid, "\n");
}

void Hessian::find_omega_batch(FILE* fid)
{
  size_t dim = num_basis * 3;
  std::vector<double> W(dim * num_kpoints);
  eig_hermitian_Jacobi_batch(dim, num_kpoints, DR.data(), DI.data(), W.data());
  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * dim;
    fprintf(fid, "%.6f ", kpath[nk]);
    for (size_t n = 0; n < dim; ++n) {
      fprintf(fid, "%g ", W[offset + n] * natural_to_THz);
    }
    fprintf(fid, "\n");
  }
}

void Hessian::find_dispersion(const Box& box, Atom& atom)
{
  FILE* fid_omega2 = fopen("omega2.out", "w");
  fprintf(fid_omega2, "#");
  for (int i = 0; i < kpath_sym.size(); ++i) {
    fprintf(fid_omega2, " %.6f", kpath_sym[i]);
  }
  fprintf(fid_omega2, " ");
  for (int i = 0; i < hsp_names.size(); ++i) {
    if (i > 0)
      fprintf(fid_omega2, "|");
    fprintf(fid_omega2, "%s", hsp_names[i].c_str());
  }
  fprintf(fid_omega2, "\n");

  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t nb = 0; nb < num_basis; ++nb) {
      size_t n1 = basis[nb];
      size_t label_1 = label[n1];
      double mass_1 = mass[label_1];
      for (size_t n2 = 0; n2 < atom.number_of_atoms; ++n2) {
        if (is_too_far(box, atom.cpu_position_per_atom, n1, n2))
          continue;
        double cos_kr, sin_kr;
        find_exp_ikr(n1, n2, kpoints.data() + nk * 3, box, atom.cpu_position_per_atom, cos_kr, sin_kr);

        size_t label_2 = label[n2];
        double mass_2 = mass[label_2];
        double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
        double* H12 = H.data() + (nb * atom.number_of_atoms + n2) * 9;
        for (size_t a = 0; a < 3; ++a) {
          for (size_t b = 0; b < 3; ++b) {
            size_t a3b = a * 3 + b;
            size_t row = label_1 * 3 + a;
            size_t col = label_2 * 3 + b;
            // cuSOLVER requires column-major
            size_t index = offset + col * num_basis * 3 + row;
            DR[index] += H12[a3b] * cos_kr * mass_factor;
            DI[index] += H12[a3b] * sin_kr * mass_factor;
          }
        }
      }
    }
    if (num_basis > 10) {
      find_omega(fid_omega2, offset, nk);
    } // > 32x32
  }
  output_D();
  if (num_basis <= 10) {
    find_omega_batch(fid_omega2);
  } // <= 32x32
  fclose(fid_omega2);
}

void Hessian::find_D(const Box& box, Atom& atom)
{
  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    size_t label_1 = label[n1];
    double mass_1 = mass[label_1];
    for (size_t n2 = 0; n2 < atom.number_of_atoms; ++n2) {
      if (is_too_far(box, atom.cpu_position_per_atom, n1, n2)) {
        continue;
      }

      size_t label_2 = label[n2];
      double mass_2 = mass[label_2];
      double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
      double* H12 = H.data() + (nb * atom.number_of_atoms + n2) * 9;
      for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
          size_t a3b = a * 3 + b;
          size_t row = label_1 * 3 + a;
          size_t col = label_2 * 3 + b;
          // cuSOLVER requires column-major
          size_t index = col * num_basis * 3 + row;
          DR[index] += H12[a3b] * mass_factor;
        }
      }
    }
  }
}

void Hessian::find_eigenvectors()
{
  std::ofstream eigfile;
  eigfile.open("eigenvector.out", std::ios::out | std::ios::binary);

  size_t dim = num_basis * 3;
  std::vector<double> W(dim);
  std::vector<double> eigenvectors(dim * dim);
  eigenvectors_symmetric_Jacobi(dim, DR.data(), W.data(), eigenvectors.data());

  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);

  // output eigenvalues
  float om2;
  for (size_t n = 0; n < dim; n++) {
    om2 = (float)(W[n] * natural_to_THz);
    eigfile.write((char*)&om2, sizeof(float));
  }

  // output eigenvectors
  float eig;
  for (size_t col = 0; col < dim; col++) {
    for (size_t a = 0; a < 3; a++) {
      for (size_t b = 0; b < num_basis; b++) {
        size_t row = a + b * 3;
        // column-major order from cuSolver
        eig = (float)eigenvectors[row + col * dim];
        eigfile.write((char*)&eig, sizeof(float));
      }
    }
  }
  eigfile.close();
}

void Hessian::parse(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("compute_phonon should have 2 parameters.\n");
  }

  if (!is_valid_real(param[1], &displacement)) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be a number.\n");
  }
  if (displacement <= 0) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be positive.\n");
  }
  printf("displacement for compute_phonon = %g A.\n", displacement);
}
