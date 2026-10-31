/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace analytic_phonon {

using Vec = std::array<double, 3>;
using Counts = std::array<int, 3>;
constexpr double pi = 3.14159265358979323846;

struct Cell {
  std::array<double, 9> h, inverse;
  Counts periodic;
};

inline Vec multiply(const std::array<double, 9>& a, const Vec& x)
{
  Vec y{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) y[i] += a[3*i+j] * x[j];
  return y;
}

inline Vec minimum_image(const Cell& cell, Vec x)
{
  x = multiply(cell.inverse, x);
  for (int d = 0; d < 3; ++d)
    if (cell.periodic[d]) x[d] -= std::nearbyint(x[d]);
  return multiply(cell.h, x);
}

inline size_t cell_count(const Counts& counts)
{
  size_t n = 1;
  for (int c : counts) {
    if (c <= 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()) / c)
      throw std::runtime_error("supercell must contain three positive, bounded integers");
    n *= c;
  }
  return n;
}

inline Counts parse_counts(std::string text)
{
  if (std::count(text.begin(), text.end(), ',') != 2)
    throw std::runtime_error("supercell syntax is nx,ny,nz");
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream input(text);
  Counts counts{};
  std::string extra;
  if (!(input >> counts[0] >> counts[1] >> counts[2]) || input >> extra)
    throw std::runtime_error("supercell syntax is nx,ny,nz");
  cell_count(counts);
  return counts;
}

inline Counts cell_index(size_t index, const Counts& c)
{
  Counts result{};
  result[2] = index % c[2]; index /= c[2];
  result[1] = index % c[1]; index /= c[1];
  result[0] = index;
  return result;
}

inline size_t validate_mapping(
  const Cell& cell, const Counts& counts, const std::vector<double>& position,
  const std::vector<double>& mass, const std::vector<int>& types)
{
  const size_t n = mass.size(), copies = cell_count(counts);
  if (!n || n % copies || position.size() != 3*n || types.size() != n)
    throw std::runtime_error("atom count is incompatible with the specified supercell");
  for (double x : cell.h)
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite phonon cell");
  for (double x : cell.inverse)
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite inverse phonon cell");
  for (int d = 0; d < 3; ++d)
    if (!cell.periodic[d] && counts[d] != 1)
      throw std::runtime_error("supercell reduction requires periodicity in replicated directions");
  for (double x : position)
    if (!std::isfinite(x)) throw std::runtime_error("nonfinite phonon coordinate");
  for (double m : mass)
    if (!std::isfinite(m) || m <= 0) throw std::runtime_error("phonon masses must be positive");
  const size_t basis = n / copies;
  for (size_t atom = 0; atom < n; ++atom) {
    const size_t b = atom % basis;
    const Counts index = cell_index(atom / basis, counts);
    Vec fractional{};
    for (int d = 0; d < 3; ++d) fractional[d] = double(index[d]) / counts[d];
    const Vec shift = multiply(cell.h, fractional);
    Vec difference{};
    for (int d = 0; d < 3; ++d)
      difference[d] = position[atom+d*n] - position[b+d*n] - shift[d];
    difference = minimum_image(cell, difference);
    for (double x : difference)
      if (std::abs(x) > 1e-5)
        throw std::runtime_error("supercell positions/order do not repeat the reference cell");
    if (types[atom] != types[b] || std::abs(mass[atom]-mass[b]) > 1e-10)
      throw std::runtime_error("supercell types/masses do not repeat the reference cell");
  }
  return basis;
}

struct KPoint {
  Vec fractional{};
  double distance = 0;
};

struct KPath {
  std::vector<KPoint> points;
  std::vector<double> ticks;
  std::vector<std::string> labels;
};

inline Vec cartesian_k(const Cell& cell, const Counts& counts, const Vec& fractional)
{
  Vec k{};
  for (int a = 0; a < 3; ++a)
    for (int d = 0; d < 3; ++d)
      k[d] += 2*pi*counts[a]*cell.inverse[3*a+d]*fractional[a];
  return k;
}

inline KPath read_path(
  const std::string& filename, int intervals, const Cell& cell, const Counts& counts)
{
  if (intervals < 1) throw std::runtime_error("kpoint_intervals must be positive");
  std::ifstream input(filename);
  if (!input) throw std::runtime_error("cannot open kpoints file: " + filename);
  KPath path;
  Vec previous{};
  bool segment = false;
  double distance = 0;
  std::string line;
  while (std::getline(input, line)) {
    const size_t comment = line.find('#');
    const std::string content = line.substr(0, comment);
    std::istringstream row(content);
    if (content.find_first_not_of(" \t\r") == std::string::npos) {
      if (comment == std::string::npos) segment = false;
      continue;
    }
    Vec next{};
    std::string label, extra;
    if (!(row >> next[0] >> next[1] >> next[2] >> label) || row >> extra)
      throw std::runtime_error("each kpoints line requires kx ky kz label");
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(next[d])) throw std::runtime_error("nonfinite kpoint");
      if (!cell.periodic[d] && std::abs(next[d]) > 1e-12)
        throw std::runtime_error("nonzero kpoint in a nonperiodic direction");
    }
    if (!segment) {
      path.points.push_back({next, distance});
    } else {
      Vec delta{};
      for (int d = 0; d < 3; ++d) delta[d] = next[d] - previous[d];
      const Vec dk = cartesian_k(cell, counts, delta);
      const double length = std::sqrt(dk[0]*dk[0]+dk[1]*dk[1]+dk[2]*dk[2]);
      for (int s = 1; s <= intervals; ++s) {
        Vec point{};
        for (int d = 0; d < 3; ++d) point[d] = previous[d] + delta[d]*s/intervals;
        path.points.push_back({point, distance + length*s/intervals});
      }
      distance += length;
    }
    path.ticks.push_back(distance);
    path.labels.push_back(label);
    previous = next;
    segment = true;
  }
  if (path.points.empty()) throw std::runtime_error("kpoints file is empty");
  return path;
}

inline void validate_wavevectors(
  const KPath& path, const Cell& cell, const Counts& counts, double interaction_range)
{
  for (const auto& point : path.points) {
    for (int d = 0; d < 3; ++d) {
      if (!cell.periodic[d]) continue;
      const double phase = point.fractional[d] * counts[d];
      const double inverse_thickness = std::sqrt(
        cell.inverse[3*d]*cell.inverse[3*d] +
        cell.inverse[3*d+1]*cell.inverse[3*d+1] +
        cell.inverse[3*d+2]*cell.inverse[3*d+2]);
      if (std::abs(phase - std::nearbyint(phase)) > 1e-10 &&
          1.0/inverse_thickness <= 2*interaction_range*(1+1e-10))
        throw std::runtime_error(
          "supercell too small for noncommensurate kpoints: periodic Hessian aliases images; "
          "increase the supercell or use commensurate kpoints");
    }
  }
}

// Average equivalent origins before Fourier interpolation. The input uses SoA;
// the reduced blocks retain the legacy [basis][supercell atom][alpha][beta] order.
inline std::vector<double> fold(
  const std::vector<double>& hessian, size_t n, const Counts& counts)
{
  const size_t copies = cell_count(counts);
  if (!n || n % copies || n > static_cast<size_t>(std::numeric_limits<int>::max())/3)
    throw std::runtime_error("invalid Cartesian Hessian dimensions");
  const size_t basis = n/copies, dim = 3*n;
  if (dim > std::numeric_limits<size_t>::max()/dim || hessian.size() != dim*dim)
    throw std::runtime_error("invalid Cartesian Hessian dimensions");
  std::vector<double> blocks(basis*n*9, 0);
  for (size_t i = 0; i < n; ++i) {
    const Counts ci = cell_index(i/basis, counts);
    for (size_t j = 0; j < n; ++j) {
      const Counts cj = cell_index(j/basis, counts);
      size_t relative = 0;
      for (int d = 0; d < 3; ++d)
        relative = relative*counts[d] + (cj[d]-ci[d]+counts[d]) % counts[d];
      const size_t offset = ((i%basis)*n + relative*basis + j%basis)*9;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const double value = hessian[(i+a*n)*dim + j+b*n];
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite Hessian");
          blocks[offset+3*a+b] += value / copies;
        }
    }
  }
  return blocks;
}

inline void dynamical_matrix(
  const std::vector<double>& blocks, const std::vector<double>& position,
  const std::vector<double>& mass, const Cell& cell, const Counts& counts,
  const Vec& fractional, std::vector<double>& real, std::vector<double>& imaginary)
{
  const size_t n = mass.size(), basis = n/cell_count(counts), dim = 3*basis;
  real.assign(dim*dim, 0); imaginary.assign(dim*dim, 0);
  const Vec k = cartesian_k(cell, counts, fractional);
  for (size_t i = 0; i < basis; ++i) {
    for (size_t j = 0; j < n; ++j) {
      Vec r{};
      for (int d = 0; d < 3; ++d) r[d] = position[j+d*n]-position[i+d*n];
      r = minimum_image(cell, r);
      const double phase = k[0]*r[0]+k[1]*r[1]+k[2]*r[2];
      const double factor = 1/std::sqrt(mass[i]*mass[j]);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          const size_t target = (3*(j%basis)+b)*dim + 3*i+a;
          const double value = blocks[(i*n+j)*9+3*a+b]*factor;
          real[target] += value*std::cos(phase);
          imaginary[target] += value*std::sin(phase);
        }
    }
  }
  double scale = 1, defect = 0;
  for (size_t i = 0; i < dim; ++i)
    for (size_t j = 0; j < dim; ++j) {
      const size_t a = i+j*dim, b = j+i*dim;
      scale = std::max(scale, std::hypot(real[a], imaginary[a]));
      defect = std::max(defect, std::hypot(real[a]-real[b], imaginary[a]+imaginary[b]));
    }
  if (!std::isfinite(scale) || !std::isfinite(defect) || defect > 1e-8*scale)
    throw std::runtime_error("dynamical matrix is not Hermitian; check the supercell mapping");
  for (size_t i = 0; i < dim; ++i)
    for (size_t j = i; j < dim; ++j) {
      const size_t a = i+j*dim, b = j+i*dim;
      const double r = 0.5*(real[a]+real[b]), im = 0.5*(imaginary[a]-imaginary[b]);
      real[a] = real[b] = r;
      imaginary[a] = im; imaginary[b] = -im;
    }
}

} // namespace analytic_phonon
