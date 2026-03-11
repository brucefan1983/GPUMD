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
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <vector>

#define M_PI 3.14159265358979323846
namespace
{
// Helper structures for automatic k-point generation
struct Vec3 {
  double x, y, z;
};

struct Mat3 {
  double data[3][3];
};

double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

Vec3 cross(const Vec3& a, const Vec3& b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Mat3 transpose(const Mat3& m)
{
  Mat3 t;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      t.data[j][i] = m.data[i][j];
  return t;
}

Vec3 matvec(const Mat3& m, const Vec3& v)
{
  return {
    m.data[0][0] * v.x + m.data[0][1] * v.y + m.data[0][2] * v.z,
    m.data[1][0] * v.x + m.data[1][1] * v.y + m.data[1][2] * v.z,
    m.data[2][0] * v.x + m.data[2][1] * v.y + m.data[2][2] * v.z};
}

Vec3 lerp(const Vec3& a, const Vec3& b, double t)
{
  return {a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z)};
}

Vec3 operator*(const Vec3& v, double s) { return {v.x * s, v.y * s, v.z * s}; }

Mat3 reciprocal_lattice(const Vec3 lat[3])
{
  double volume = dot(lat[0], cross(lat[1], lat[2]));
  double factor = 2.0 * M_PI / volume;

  Vec3 c0 = cross(lat[1], lat[2]) * factor;
  Vec3 c1 = cross(lat[2], lat[0]) * factor;
  Vec3 c2 = cross(lat[0], lat[1]) * factor;
  Mat3 rec = {{{c0.x, c0.y, c0.z}, {c1.x, c1.y, c1.z}, {c2.x, c2.y, c2.z}}};
  return transpose(rec);
}
} // namespace

void Hessian::compute(
  Force& force,
  Box& box,
  const std::vector<double>& cpu_mass,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  initialize(cpu_mass, box, force, type.size());
  find_H(
    force,
    box,
    cpu_position_per_atom,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

  if (num_kpoints == 1) // currently for Alex's GKMA calculations
  {
    find_D(box, cpu_position_per_atom);
    find_eigenvectors();
  } else {
    find_dispersion(box, cpu_position_per_atom);
  }
}

void Hessian::get_cutoff_from_potential(Force& force)
{
  for (const auto& potential : force.potentials) {
      cutoff = potential->rc;
  }
  phonon_cutoff = cutoff * 2.0;
  printf("Using cutoff for phonon calculations: %g A.\n", phonon_cutoff);
}

void Hessian::create_basis(const std::vector<double>& cpu_mass, size_t N)
{
  num_basis = N / (cxyz[0] * cxyz[1] * cxyz[2]);

  basis.resize(num_basis);
  mass.resize(num_basis);
  for (size_t i = 0; i < num_basis; ++i) {
    basis[i] = i;
    mass[i] = cpu_mass[i];
  }

  label.resize(N);
  for (size_t n = 0; n < N; ++n) {
    size_t atom = n % num_basis;
    label[n] = atom;
  }
}

void Hessian::create_kpoints(const Box& box)
{
  std::ifstream kin("kpoints.in");
  if (!kin)
    PRINT_INPUT_ERROR("Cannot open kpoints.in file.");

  std::vector<std::vector<Vec3>> hsps;
  std::vector<Vec3> hsp;
  sym_names.clear();
  std::string line;
  std::string names;

  while (std::getline(kin, line)) {
    const auto beg = line.find_first_not_of(" \t\r\n");
    if (beg == std::string::npos) {
      if (!hsp.empty()) {
        hsps.push_back(hsp);
        sym_names.push_back(names);
        names.clear();
        hsp.clear();
      }
      continue;
    }
    if (line[beg] == '#')
      continue;

    double x, y, z;
    char name[16];
    int n = sscanf(line.c_str(), "%lf %lf %lf %15[^# \n]", &x, &y, &z, name);
    if (n < 4) {
      PRINT_INPUT_ERROR("kpoints.in file format error.");
    }

    hsp.push_back({x, y, z});
    if (!names.empty())
      names += " ";
    names += name;
  }
  if (!hsp.empty())
    hsps.push_back(hsp);
  sym_names.push_back(names);

  if (!sym_names.empty() && sym_names.back().empty()) {
    sym_names.pop_back();
  }

  num_kpoints = 1 - hsps.size();
  for (const auto& seg : hsps)
    num_kpoints += seg.size();
  kpath_sym.resize(num_kpoints);
  num_kpoints = (num_kpoints - 1) * 100 + 1;

  const Vec3 origin_lattice[3] = {
    {box.cpu_h[0] / cxyz[0], box.cpu_h[3] / cxyz[0], box.cpu_h[6] / cxyz[0]},
    {box.cpu_h[1] / cxyz[1], box.cpu_h[4] / cxyz[1], box.cpu_h[7] / cxyz[1]},
    {box.cpu_h[2] / cxyz[2], box.cpu_h[5] / cxyz[2], box.cpu_h[8] / cxyz[2]}};
  const auto rec_lat = reciprocal_lattice(origin_lattice);

  kpoints.resize(num_kpoints * 3);
  kpath.resize(num_kpoints);
  std::vector<double> sym_idx;

  size_t k_idx = 0;
  double kpath_len = 0.0;
  auto k_first = matvec(rec_lat, hsps[0][0]);
  kpoints[0] = k_first.x;
  kpoints[1] = k_first.y;
  kpoints[2] = k_first.z;
  kpath[k_idx] = kpath_len;
  sym_idx.push_back(k_idx);
  ++k_idx;

  for (const auto& hsp : hsps) {
    for (size_t i = 1; i < hsp.size(); ++i) {
      const auto& start = matvec(rec_lat, hsp[i - 1]);
      const auto& end = matvec(rec_lat, hsp[i]);

      for (int j = 1; j <= 100; ++j) {
        double t = j * 0.01;
        auto kpt = lerp(start, end, t);
        kpoints[k_idx * 3 + 0] = kpt.x;
        kpoints[k_idx * 3 + 1] = kpt.y;
        kpoints[k_idx * 3 + 2] = kpt.z;
        
        double dx, dy, dz;
        if (i == 1 && j == 1){
          dx = kpt.x - start.x;
          dy = kpt.y - start.y;
          dz = kpt.z - start.z;
        } else{
          dx = kpt.x - kpoints[k_idx * 3 - 3];
          dy = kpt.y - kpoints[k_idx * 3 - 2];
          dz = kpt.z - kpoints[k_idx * 3 - 1];
        }
        kpath_len += std::sqrt(dx * dx + dy * dy + dz * dz);
        kpath[k_idx] = kpath_len;

        if (j == 100)
          sym_idx.push_back(k_idx);
        ++k_idx;
      }
    }
  }

  for (size_t kp = 0; kp < kpath_sym.size(); ++kp) {
    kpath_sym[kp] = kpath[sym_idx[kp]];
  }
}

void Hessian::initialize(
  const std::vector<double>& cpu_mass, Box& box, Force& force, size_t N)
{
  get_cutoff_from_potential(force);

  std::ifstream fin("run.in");
  std::string line;
  bool f_rep = false;
  while (std::getline(fin, line)) {
    auto tokens = get_tokens(line);
    if (!tokens.empty() && tokens[0][0] != '#' && tokens[0] == "replicate") {  // 跳过空行和注释行
      f_rep = true;
      cxyz[0] = get_int_from_token(tokens[1], __FILE__, __LINE__);
      cxyz[1] = get_int_from_token(tokens[2], __FILE__, __LINE__);
      cxyz[2] = get_int_from_token(tokens[3], __FILE__, __LINE__);
    }
    break;
  }
  fin.close();
  if (!f_rep) {
    PRINT_INPUT_ERROR("replicate keyword not found in run.in file.");
  }

  int s_c[3] = {1, 1, 1};
  int stru_pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  double volume = box.get_volume();
  for (int i= 0; i < 3; ++i){
    double thickness = volume / box.get_area(i);
    double ori_thick = thickness / cxyz[i];
    printf("thickness in %d direction: %f\n", i, ori_thick);
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
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, cpu_position_per_atom, n1, n2)) {
        continue;
      }
      size_t offset = (nb * number_of_atoms + n2) * 9;
      find_H12(
        displacement,
        n1,
        n2,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom,
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

void Hessian::find_dispersion(const Box& box, const std::vector<double>& cpu_position_per_atom)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;

  FILE* fid_omega2 = fopen("omega2.out", "w");
  fprintf(fid_omega2, "#");
  for (size_t i = 0; i < kpath_sym.size(); ++i) {
    fprintf(fid_omega2, " %.6f", kpath_sym[i]);
  }
  fprintf(fid_omega2, " ");
  for (size_t i = 0; i < sym_names.size(); ++i) {
    if (i > 0)
      fprintf(fid_omega2, "|");
    fprintf(fid_omega2, "%s", sym_names[i].c_str());
  }
  fprintf(fid_omega2, "\n");

  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t nb = 0; nb < num_basis; ++nb) {
      size_t n1 = basis[nb];
      size_t label_1 = label[n1];
      double mass_1 = mass[label_1];
      for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
        if (is_too_far(box, cpu_position_per_atom, n1, n2))
          continue;
        double cos_kr, sin_kr;
        find_exp_ikr(n1, n2, kpoints.data() + nk * 3, box, cpu_position_per_atom, cos_kr, sin_kr);

        size_t label_2 = label[n2];
        double mass_2 = mass[label_2];
        double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
        double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
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

void Hessian::find_D(const Box& box, std::vector<double>& cpu_position_per_atom)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    size_t label_1 = label[n1];
    double mass_1 = mass[label_1];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, cpu_position_per_atom, n1, n2)) {
        continue;
      }

      size_t label_2 = label[n2];
      double mass_2 = mass[label_2];
      double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
      double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
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

void Hessian::parse(const char** param, size_t num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("compute_phonon should have 2 parameters.\n");
  }

  // displacement
  if (!is_valid_real(param[1], &displacement)) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be a number.\n");
  }
  if (displacement <= 0) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be positive.\n");
  }
  printf("displacement for compute_phonon = %g A.\n", displacement);
}
