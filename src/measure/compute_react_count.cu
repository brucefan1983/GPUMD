/*
    Copyright 2026 GPUMD development team
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

/*----------------------------------------------------------------------------80
compute_react_count: distance-based transient-bond species, bond, coordination
and reaction-event statistics for variable-topology MD.

The file has two parts:

  1. A host-only, dependency-free core (react_count namespace).  It is the
     complete analysis: cell list, bond perception, molecule labelling,
     formula aggregation, event balancing and restart sidecar I/O.
  2. A thin GPUMD Action adapter.  It parses the run.in command, copies the
     current positions/types to the host and calls the core.

The split keeps the algorithm testable on a CPU (compile with
-DREACT_COUNT_CORE_ONLY) while the production path is a normal GPUMD
compute action.  No extra header, no CUDA kernel, no MPI.
------------------------------------------------------------------------------*/

#ifndef REACT_COUNT_CORE_ONLY
#include "action.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace react_count
{

constexpr int MAXELEMS = 32;

struct DefaultElement {
  const char* symbol;
  double radius;
  int valence;
};

// The same distance-perception table used by the LAMMPS fix and by
// ReaxTools.  Unknown elements have no radius and valence 4.
static const DefaultElement DEFAULT_ELEMENTS[] = {
  {"H", 1.05, 1},  {"He", 1.40, 0}, {"Li", 1.82, 1}, {"Be", 2.00, 2}, {"B", 2.00, 3},
  {"C", 1.70, 4},  {"N", 1.55, 4},  {"O", 1.70, 2},  {"F", 1.35, 1},  {"Ne", 1.54, 0},
  {"Na", 1.36, 1}, {"Mg", 1.18, 2}, {"Al", 2.00, 3}, {"Si", 2.10, 4}, {"P", 1.80, 6},
  {"S", 1.80, 6},  {"Cl", 2.27, 7}, {"Ar", 1.88, 0}, {"K", 1.76, 1},  {"Ca", 1.37, 2},
  {"Fe", 2.00, 6}, {"Co", 2.00, 3}, {"Ni", 1.63, 2}, {"Cu", 1.90, 2}, {"Zn", 1.39, 2},
  {"Br", 1.85, 7}, {"I", 1.98, 7},
};

inline void fail(const std::string& message)
{
  std::fprintf(stderr, "compute_react_count: %s\n", message.c_str());
  std::exit(1);
}

inline std::vector<std::string> split_commas(const std::string& text)
{
  std::vector<std::string> out;
  std::string item;
  for (char c : text) {
    if (c == ',') {
      if (!item.empty())
        out.push_back(item);
      item.clear();
    } else if (c != ' ' && c != '\t') {
      item.push_back(c);
    }
  }
  if (!item.empty())
    out.push_back(item);
  return out;
}

inline bool parse_bool(const std::string& text, bool& value)
{
  if (text == "yes" || text == "true" || text == "1") {
    value = true;
    return true;
  }
  if (text == "no" || text == "false" || text == "0") {
    value = false;
    return true;
  }
  return false;
}

struct Config {
  std::string output_base;
  std::string restart_file;
  int nfreq = 1;
  double cutoff = 1.2;
  int max_neigh = 10;
  std::vector<std::string> elements; // type 0 .. ntypes-1
  std::vector<double> radius;
  std::vector<int> valence;
  std::vector<std::string> order{"C", "H", "N", "O"};
  bool events = false;
  bool require_balance = false;
  bool offset_steps = false; // adapter: continue step numbering from the sidecar
  int perception = 0;        // 0 = reax (tag-order greedy), 1 = mutual nearest

  void make_elements(
    const std::vector<std::string>& symbols,
    const std::map<std::string, double>& radius_override,
    const std::map<std::string, int>& valence_override)
  {
    elements = symbols;
    radius.assign(elements.size(), 0.0);
    valence.assign(elements.size(), 4);
    for (size_t t = 0; t < elements.size(); ++t) {
      const std::string& sym = elements[t];
      for (const auto& entry : DEFAULT_ELEMENTS) {
        if (sym == entry.symbol) {
          radius[t] = entry.radius;
          valence[t] = entry.valence;
          break;
        }
      }
      auto rit = radius_override.find(sym);
      if (rit != radius_override.end())
        radius[t] = rit->second;
      auto vit = valence_override.find(sym);
      if (vit != valence_override.end())
        valence[t] = vit->second;
    }
  }

  std::string element_of(int type) const
  {
    if (type >= 0 && type < static_cast<int>(elements.size()) && !elements[type].empty())
      return elements[type];
    return std::to_string(type);
  }

  int max_valence() const
  {
    int value = 0;
    for (int v : valence)
      value = std::max(value, v);
    return value;
  }
};

struct Box {
  double h[18]{};
  int pbc[3]{1, 1, 1};
  bool orth = true;
  double thickness[3]{1.0, 1.0, 1.0};

  void set_lattice(const double lattice[9], int px, int py, int pz)
  {
    // GPUMD stores box vectors as columns: a=(h0,h3,h6), ...
    h[0] = lattice[0];
    h[3] = lattice[1];
    h[6] = lattice[2];
    h[1] = lattice[3];
    h[4] = lattice[4];
    h[7] = lattice[5];
    h[2] = lattice[6];
    h[5] = lattice[7];
    h[8] = lattice[8];
    pbc[0] = px;
    pbc[1] = py;
    pbc[2] = pz;
    make_inverse();
    orth = (h[1] == 0.0 && h[2] == 0.0 && h[3] == 0.0 && h[5] == 0.0 && h[6] == 0.0 && h[7] == 0.0);
    make_thickness();
  }

  void set_from_gpumd(const double h18[18], int px, int py, int pz)
  {
    for (int i = 0; i < 9; ++i)
      h[i] = h18[i];
    pbc[0] = px;
    pbc[1] = py;
    pbc[2] = pz;
    make_inverse();
    orth = (h[1] == 0.0 && h[2] == 0.0 && h[3] == 0.0 && h[5] == 0.0 && h[6] == 0.0 && h[7] == 0.0);
    make_thickness();
  }

  void make_inverse()
  {
    h[9] = h[4] * h[8] - h[5] * h[7];
    h[10] = h[2] * h[7] - h[1] * h[8];
    h[11] = h[1] * h[5] - h[2] * h[4];
    h[12] = h[5] * h[6] - h[3] * h[8];
    h[13] = h[0] * h[8] - h[2] * h[6];
    h[14] = h[2] * h[3] - h[0] * h[5];
    h[15] = h[3] * h[7] - h[4] * h[6];
    h[16] = h[1] * h[6] - h[0] * h[7];
    h[17] = h[0] * h[4] - h[1] * h[3];
    const double det = h[0] * (h[4] * h[8] - h[5] * h[7]) + h[1] * (h[5] * h[6] - h[3] * h[8]) +
                       h[2] * (h[3] * h[7] - h[4] * h[6]);
    if (det == 0.0)
      fail("box has zero volume");
    for (int i = 9; i < 18; ++i)
      h[i] /= det;
  }

  void make_thickness()
  {
    const double a[3] = {h[0], h[3], h[6]};
    const double b[3] = {h[1], h[4], h[7]};
    const double c[3] = {h[2], h[5], h[8]};
    auto cross = [](const double u[3], const double v[3], double out[3]) {
      out[0] = u[1] * v[2] - u[2] * v[1];
      out[1] = u[2] * v[0] - u[0] * v[2];
      out[2] = u[0] * v[1] - u[1] * v[0];
    };
    auto norm = [](const double u[3]) {
      return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    };
    double bc[3], ca[3], ab[3];
    cross(b, c, bc);
    cross(c, a, ca);
    cross(a, b, ab);
    const double volume = std::fabs(a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]);
    if (volume == 0.0)
      fail("box has zero volume");
    thickness[0] = volume / norm(bc);
    thickness[1] = volume / norm(ca);
    thickness[2] = volume / norm(ab);
  }

  void mic(double& x12, double& y12, double& z12) const
  {
    if (orth) {
      const double Lx = h[0], Ly = h[4], Lz = h[8];
      if (pbc[0]) {
        if (x12 < -0.5 * Lx)
          x12 += Lx;
        else if (x12 > 0.5 * Lx)
          x12 -= Lx;
      }
      if (pbc[1]) {
        if (y12 < -0.5 * Ly)
          y12 += Ly;
        else if (y12 > 0.5 * Ly)
          y12 -= Ly;
      }
      if (pbc[2]) {
        if (z12 < -0.5 * Lz)
          z12 += Lz;
        else if (z12 > 0.5 * Lz)
          z12 -= Lz;
      }
    } else {
      double sx = h[9] * x12 + h[10] * y12 + h[11] * z12;
      double sy = h[12] * x12 + h[13] * y12 + h[14] * z12;
      double sz = h[15] * x12 + h[16] * y12 + h[17] * z12;
      if (pbc[0])
        sx -= std::nearbyint(sx);
      if (pbc[1])
        sy -= std::nearbyint(sy);
      if (pbc[2])
        sz -= std::nearbyint(sz);
      x12 = h[0] * sx + h[1] * sy + h[2] * sz;
      y12 = h[3] * sx + h[4] * sy + h[5] * sz;
      z12 = h[6] * sx + h[7] * sy + h[8] * sz;
    }
  }
};

struct SpecInfo {
  std::string formula;
  std::map<std::string, int> elems;
  int atoms = 0;
};

struct MoleculeRec {
  uint64_t hash = 0;
  int count = 0;
  std::array<int, MAXELEMS> elems{};
};

inline uint64_t tag_hash(uint64_t tag)
{
  uint64_t z = tag + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

class Engine
{
public:
  explicit Engine(Config config) : cfg_(std::move(config)) {}

  ~Engine() { close(); }

  void open(int number_of_atoms)
  {
    n_ = number_of_atoms;
    ntypes_ = static_cast<int>(cfg_.elements.size());
    if (ntypes_ <= 0)
      fail("no elements are defined");
    if (ntypes_ > MAXELEMS)
      fail("too many atom types");

    maxbond_ = std::max(1, cfg_.max_valence());
    maxbond_ = std::max(maxbond_, cfg_.max_neigh);

    rc_ = 0.0;
    for (int a = 0; a < ntypes_; ++a) {
      for (int b = 0; b < ntypes_; ++b) {
        if (cfg_.radius[a] <= 0.0 || cfg_.radius[b] <= 0.0)
          continue;
        rc_ = std::max(rc_, 0.5 * (cfg_.radius[a] + cfg_.radius[b]) * cfg_.cutoff);
      }
    }
    if (rc_ <= 0.0)
      fail("no valid element radii; use the radius keyword");
    // Candidate margin protects the exact float-threshold test from cell
    // boundary roundoff.  The exact test below still uses the physical bond
    // radius, so the margin only affects the candidate list.
    rc_ += 0.05;

    nbond_.assign(n_, 0);
    bond_idx_.assign(static_cast<size_t>(n_) * maxbond_, -1);
    parent_.assign(n_, 0);
    cur_hash_.assign(n_, 0);
    prev_hash_.assign(n_, 0);
    cell_of_.assign(n_, 0);
    cell_x_.assign(n_, 0);
    cell_y_.assign(n_, 0);
    cell_z_.assign(n_, 0);
    selected_.assign(n_, {});

    open_output();
    read_state();
  }

  void close()
  {
    if (fspecies_)
      std::fclose(fspecies_);
    if (fbond_)
      std::fclose(fbond_);
    if (fcoord_)
      std::fclose(fcoord_);
    if (fevents_)
      std::fclose(fevents_);
    fspecies_ = fbond_ = fcoord_ = fevents_ = nullptr;
  }

  void step(long long step, int number_of_atoms, const int* type, const double* pos, const Box& box)
  {
    if (number_of_atoms != n_)
      fail("atom count changed during the run");
    type_ = type;
    pos_ = pos;
    box_ = &box;

    const long long output_step = cfg_.offset_steps ? step + step_offset_ : step;
    last_step_ = output_step;

    build_topology();
    label_molecules();
    build_records();
    write_counts(output_step);
    do_events(output_step);
    if (cfg_.events && !cfg_.restart_file.empty())
      write_state();

    if (fspecies_)
      std::fflush(fspecies_);
    if (fbond_)
      std::fflush(fbond_);
    if (fcoord_)
      std::fflush(fcoord_);
    if (fevents_)
      std::fflush(fevents_);
  }

private:
  struct Candidate {
    float rel;
    int index;
  };

  Config cfg_;
  int n_ = 0;
  int ntypes_ = 0;
  int maxbond_ = 1;
  double rc_ = 0.0;

  const int* type_ = nullptr;
  const double* pos_ = nullptr;
  const Box* box_ = nullptr;

  FILE* fspecies_ = nullptr;
  FILE* fbond_ = nullptr;
  FILE* fcoord_ = nullptr;
  FILE* fevents_ = nullptr;

  std::vector<int> nbond_;
  std::vector<int> bond_idx_;
  std::vector<int> parent_;
  std::vector<int> cell_of_;
  std::vector<int> cell_x_;
  std::vector<int> cell_y_;
  std::vector<int> cell_z_;
  std::vector<std::vector<int>> selected_;

  std::vector<uint64_t> cur_hash_;
  std::vector<uint64_t> prev_hash_;

  bool first_frame_ = true;
  int event_id_ = 0;
  long long step_offset_ = 0;
  long long last_step_ = 0;
  std::map<std::string, int> species_counts_;
  std::unordered_map<uint64_t, SpecInfo> prev_registry_;
  std::unordered_map<uint64_t, SpecInfo> curr_registry_;

  void open_output()
  {
    fspecies_ = std::fopen((cfg_.output_base + ".species.csv").c_str(), "w");
    if (!fspecies_)
      fail("cannot open " + cfg_.output_base + ".species.csv");
    fbond_ = std::fopen((cfg_.output_base + ".bond.csv").c_str(), "w");
    if (!fbond_)
      fail("cannot open " + cfg_.output_base + ".bond.csv");
    fcoord_ = std::fopen((cfg_.output_base + ".coord.csv").c_str(), "w");
    if (!fcoord_)
      fail("cannot open " + cfg_.output_base + ".coord.csv");
    if (cfg_.events) {
      fevents_ = std::fopen((cfg_.output_base + ".events.csv").c_str(), "w");
      if (!fevents_)
        fail("cannot open " + cfg_.output_base + ".events.csv");
    }
    std::fprintf(fspecies_, "step,formula,count\n");
    std::fprintf(fbond_, "step,pair,count\n");
    std::fprintf(fcoord_, "step,element,coordination,count\n");
    if (fevents_)
      std::fprintf(
        fevents_,
        "step,event_id,n_reactants,n_products,reactants,"
        "products,balanced,atom_conserved\n");
  }

  int find_root(std::vector<int>& parent, int i)
  {
    int root = i;
    while (parent[root] != root)
      root = parent[root];
    while (parent[i] != i) {
      const int next = parent[i];
      parent[i] = root;
      i = next;
    }
    return root;
  }

  void unite_min(std::vector<int>& parent, int a, int b)
  {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a == b)
      return;
    if (a < b)
      parent[b] = a;
    else
      parent[a] = b;
  }

  int nx_ = 1, ny_ = 1, nz_ = 1;
  std::vector<int> cell_count_;
  std::vector<int> cell_start_;
  std::vector<int> cell_contents_;

  void build_cells()
  {
    nx_ = box_->pbc[0] ? std::max(1, static_cast<int>(std::floor(box_->thickness[0] / rc_))) : 1;
    ny_ = box_->pbc[1] ? std::max(1, static_cast<int>(std::floor(box_->thickness[1] / rc_))) : 1;
    nz_ = box_->pbc[2] ? std::max(1, static_cast<int>(std::floor(box_->thickness[2] / rc_))) : 1;

    const int ncells = nx_ * ny_ * nz_;
    cell_count_.assign(ncells, 0);
    cell_start_.assign(ncells + 1, 0);
    cell_contents_.assign(n_, 0);

    for (int i = 0; i < n_; ++i) {
      const double* p = pos_ + 3 * i;
      const double sx = box_->h[9] * p[0] + box_->h[10] * p[1] + box_->h[11] * p[2];
      const double sy = box_->h[12] * p[0] + box_->h[13] * p[1] + box_->h[14] * p[2];
      const double sz = box_->h[15] * p[0] + box_->h[16] * p[1] + box_->h[17] * p[2];
      int ix = static_cast<int>(std::floor(sx * box_->thickness[0] / rc_));
      int iy = static_cast<int>(std::floor(sy * box_->thickness[1] / rc_));
      int iz = static_cast<int>(std::floor(sz * box_->thickness[2] / rc_));
      if (box_->pbc[0]) {
        ix %= nx_;
        if (ix < 0)
          ix += nx_;
      } else {
        ix = std::min(std::max(ix, 0), nx_ - 1);
      }
      if (box_->pbc[1]) {
        iy %= ny_;
        if (iy < 0)
          iy += ny_;
      } else {
        iy = std::min(std::max(iy, 0), ny_ - 1);
      }
      if (box_->pbc[2]) {
        iz %= nz_;
        if (iz < 0)
          iz += nz_;
      } else {
        iz = std::min(std::max(iz, 0), nz_ - 1);
      }
      cell_x_[i] = ix;
      cell_y_[i] = iy;
      cell_z_[i] = iz;
      const int c = ix + nx_ * (iy + ny_ * iz);
      cell_of_[i] = c;
      ++cell_count_[c];
    }

    int sum = 0;
    for (int c = 0; c < ncells; ++c) {
      cell_start_[c] = sum;
      sum += cell_count_[c];
    }
    cell_start_[ncells] = sum;

    std::vector<int> cursor = cell_start_;
    for (int i = 0; i < n_; ++i) {
      const int c = cell_of_[i];
      cell_contents_[cursor[c]++] = i;
    }
  }

  void collect_candidates(int i, std::vector<Candidate>& candidates) const
  {
    candidates.clear();
    const int ti = type_[i];
    const double ri = cfg_.radius[ti];
    if (ri <= 0.0)
      return;

    const double* pi = pos_ + 3 * i;
    const float xi = static_cast<float>(pi[0]);
    const float yi = static_cast<float>(pi[1]);
    const float zi = static_cast<float>(pi[2]);

    const int ix = cell_x_[i];
    const int iy = cell_y_[i];
    const int iz = cell_z_[i];

    for (int dz = -1; dz <= 1; ++dz) {
      int jz = iz + dz;
      if (box_->pbc[2]) {
        jz %= nz_;
        if (jz < 0)
          jz += nz_;
      } else if (jz < 0 || jz >= nz_) {
        continue;
      }
      for (int dy = -1; dy <= 1; ++dy) {
        int jy = iy + dy;
        if (box_->pbc[1]) {
          jy %= ny_;
          if (jy < 0)
            jy += ny_;
        } else if (jy < 0 || jy >= ny_) {
          continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
          int jx = ix + dx;
          if (box_->pbc[0]) {
            jx %= nx_;
            if (jx < 0)
              jx += nx_;
          } else if (jx < 0 || jx >= nx_) {
            continue;
          }
          const int c = jx + nx_ * (jy + ny_ * jz);
          const int begin = cell_start_[c];
          const int end = cell_start_[c + 1];
          for (int m = begin; m < end; ++m) {
            const int j = cell_contents_[m];
            if (j == i)
              continue;
            const int tj = type_[j];
            const double rj = cfg_.radius[tj];
            if (rj <= 0.0)
              continue;

            const float bond_r = 0.5f * (static_cast<float>(ri) + static_cast<float>(rj)) *
                                 static_cast<float>(cfg_.cutoff);
            const float bond_sq = bond_r * bond_r;
            const double* pj = pos_ + 3 * j;
            const float xj = static_cast<float>(pj[0]);
            const float yj = static_cast<float>(pj[1]);
            const float zj = static_cast<float>(pj[2]);
            double ddx = static_cast<double>(xi) - static_cast<double>(xj);
            double ddy = static_cast<double>(yi) - static_cast<double>(yj);
            double ddz = static_cast<double>(zi) - static_cast<double>(zj);
            box_->mic(ddx, ddy, ddz);
            const float dist_sq = static_cast<float>(ddx * ddx + ddy * ddy + ddz * ddz);
            if (dist_sq > bond_sq)
              continue;
            const float rel = dist_sq / bond_sq;
            if (rel >= 1.0f)
              continue;
            candidates.push_back({rel, j});
          }
        }
      }
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      if (a.rel != b.rel)
        return a.rel < b.rel;
      return a.index < b.index;
    });
    if (static_cast<int>(candidates.size()) > cfg_.max_neigh)
      candidates.resize(cfg_.max_neigh);
  }

  bool has_bond(int i, int j) const
  {
    for (int k = 0; k < nbond_[i]; ++k)
      if (bond_idx_[static_cast<size_t>(i) * maxbond_ + k] == j)
        return true;
    return false;
  }

  void add_bond(int i, int j)
  {
    if (nbond_[i] >= maxbond_ || nbond_[j] >= maxbond_)
      return;
    bond_idx_[static_cast<size_t>(i) * maxbond_ + nbond_[i]++] = j;
    bond_idx_[static_cast<size_t>(j) * maxbond_ + nbond_[j]++] = i;
  }

  void build_topology()
  {
    std::fill(nbond_.begin(), nbond_.end(), 0);
    std::fill(bond_idx_.begin(), bond_idx_.end(), -1);
    build_cells();

    std::vector<Candidate> candidates;
    candidates.reserve(64);

    if (cfg_.perception == 0) {
      for (int i = 0; i < n_; ++i) {
        const int vi = cfg_.valence[type_[i]];
        if (vi <= 0)
          continue;
        collect_candidates(i, candidates);
        for (const Candidate& c : candidates) {
          if (nbond_[i] >= vi)
            break;
          const int j = c.index;
          const int vj = cfg_.valence[type_[j]];
          if (vj <= 0 || nbond_[j] >= vj)
            continue;
          if (has_bond(i, j))
            continue;
          add_bond(i, j);
        }
      }
    } else {
      for (int i = 0; i < n_; ++i) {
        selected_[i].clear();
        const int vi = cfg_.valence[type_[i]];
        if (vi <= 0)
          continue;
        collect_candidates(i, candidates);
        const int nsel = std::min(vi, static_cast<int>(candidates.size()));
        selected_[i].reserve(nsel);
        for (int k = 0; k < nsel; ++k)
          selected_[i].push_back(candidates[k].index);
      }

      for (int i = 0; i < n_; ++i) {
        const int vi = cfg_.valence[type_[i]];
        if (vi <= 0)
          continue;
        for (int j : selected_[i]) {
          if (nbond_[i] >= vi)
            break;
          const int vj = cfg_.valence[type_[j]];
          if (vj <= 0 || nbond_[j] >= vj)
            continue;
          if (std::find(selected_[j].begin(), selected_[j].end(), i) == selected_[j].end())
            continue;
          if (has_bond(i, j))
            continue;
          add_bond(i, j);
        }
      }
    }
  }

  void label_molecules()
  {
    for (int i = 0; i < n_; ++i)
      parent_[i] = i;
    for (int i = 0; i < n_; ++i) {
      for (int k = 0; k < nbond_[i]; ++k) {
        const int j = bond_idx_[static_cast<size_t>(i) * maxbond_ + k];
        if (j >= 0)
          unite_min(parent_, i, j);
      }
    }
    for (int i = 0; i < n_; ++i)
      parent_[i] = find_root(parent_, i);
  }

  std::string formula_of(const std::map<std::string, int>& counts) const
  {
    std::map<std::string, int> left = counts;
    std::string out;
    for (const auto& sym : cfg_.order) {
      auto it = left.find(sym);
      if (it == left.end())
        continue;
      if (it->second > 0) {
        out += sym;
        if (it->second > 1)
          out += std::to_string(it->second);
      }
      left.erase(it);
    }
    for (const auto& kv : left) {
      if (kv.second <= 0)
        continue;
      out += kv.first;
      if (kv.second > 1)
        out += std::to_string(kv.second);
    }
    return out;
  }

  void build_records()
  {
    std::unordered_map<int, MoleculeRec> local;
    local.reserve(n_ / 2 + 1);
    for (int i = 0; i < n_; ++i) {
      const int root = parent_[i];
      MoleculeRec& rec = local[root];
      rec.count += 1;
      rec.hash ^= tag_hash(static_cast<uint64_t>(i + 1));
      rec.elems[type_[i]] += 1;
    }

    std::vector<uint64_t> root_hash(n_, 0);
    species_counts_.clear();
    curr_registry_.clear();
    curr_registry_.reserve(local.size());

    for (const auto& kv : local) {
      const MoleculeRec& rec = kv.second;
      std::map<std::string, int> elems;
      for (int t = 0; t < ntypes_; ++t) {
        if (rec.elems[t] > 0)
          elems[cfg_.element_of(t)] += rec.elems[t];
      }
      const std::string formula = formula_of(elems);
      root_hash[kv.first] = rec.hash;
      species_counts_[formula] += 1;
      curr_registry_[rec.hash] = SpecInfo{formula, elems, rec.count};
    }

    for (int i = 0; i < n_; ++i)
      cur_hash_[i] = root_hash[parent_[i]];
  }

  void write_counts(long long step)
  {
    for (const auto& kv : species_counts_)
      std::fprintf(fspecies_, "%lld,%s,%d\n", step, kv.first.c_str(), kv.second);

    const int stride = ntypes_;
    std::vector<int> pair_counts(static_cast<size_t>(stride) * stride, 0);
    for (int i = 0; i < n_; ++i) {
      const int ti = type_[i];
      for (int k = 0; k < nbond_[i]; ++k) {
        const int j = bond_idx_[static_cast<size_t>(i) * maxbond_ + k];
        if (j < 0 || i > j)
          continue;
        const int tj = type_[j];
        pair_counts[static_cast<size_t>(ti) * stride + tj] += 1;
      }
    }

    std::map<std::string, int> pairs;
    for (int a = 0; a < ntypes_; ++a) {
      for (int b = a; b < ntypes_; ++b) {
        int count = pair_counts[static_cast<size_t>(a) * stride + b];
        if (a != b)
          count += pair_counts[static_cast<size_t>(b) * stride + a];
        if (count == 0)
          continue;
        std::string x = cfg_.element_of(a);
        std::string y = cfg_.element_of(b);
        if (x < y)
          std::swap(x, y);
        pairs[x + "-" + y] += count;
      }
    }
    for (const auto& kv : pairs)
      std::fprintf(fbond_, "%lld,%s,%d\n", step, kv.first.c_str(), kv.second);

    const int ncoord = maxbond_ + 1;
    std::vector<int> coord_counts(static_cast<size_t>(ntypes_) * ncoord, 0);
    for (int i = 0; i < n_; ++i) {
      int c = nbond_[i];
      if (c > maxbond_)
        c = maxbond_;
      coord_counts[static_cast<size_t>(type_[i]) * ncoord + c] += 1;
    }

    std::vector<int> types(ntypes_);
    for (int t = 0; t < ntypes_; ++t)
      types[t] = t;
    std::sort(types.begin(), types.end(), [&](int a, int b) {
      if (cfg_.element_of(a) != cfg_.element_of(b))
        return cfg_.element_of(a) < cfg_.element_of(b);
      return a < b;
    });
    for (int t : types) {
      int maxc = cfg_.valence[t] > 0 ? cfg_.valence[t] : maxbond_;
      if (maxc > maxbond_)
        maxc = maxbond_;
      for (int c = 0; c <= maxc; ++c) {
        const int count = coord_counts[static_cast<size_t>(t) * ncoord + c];
        std::fprintf(fcoord_, "%lld,%s,%d,%d\n", step, cfg_.element_of(t).c_str(), c, count);
      }
    }
  }

  void do_events(long long step)
  {
    if (!cfg_.events)
      return;

    if (first_frame_) {
      prev_hash_ = cur_hash_;
      prev_registry_ = curr_registry_;
      first_frame_ = false;
      return;
    }

    struct Change {
      uint64_t old_hash;
      uint64_t new_hash;
    };
    std::vector<Change> changes;
    changes.reserve(n_ / 8 + 1);
    for (int i = 0; i < n_; ++i) {
      if (prev_hash_[i] != cur_hash_[i])
        changes.push_back({prev_hash_[i], cur_hash_[i]});
    }
    prev_hash_ = cur_hash_;

    if (changes.empty()) {
      prev_registry_ = curr_registry_;
      return;
    }

    std::unordered_map<uint64_t, uint64_t> uf;
    uf.reserve(changes.size() * 2);
    auto add_key = [&](uint64_t key) {
      if (uf.find(key) == uf.end())
        uf.emplace(key, key);
    };
    auto find_key = [&](uint64_t key) {
      uint64_t root = key;
      while (uf[root] != root)
        root = uf[root];
      while (uf[key] != key) {
        const uint64_t next = uf[key];
        uf[key] = root;
        key = next;
      }
      return root;
    };
    auto unite_key = [&](uint64_t a, uint64_t b) {
      a = find_key(a);
      b = find_key(b);
      if (a != b)
        uf[b] = a;
    };

    for (const Change& c : changes) {
      add_key(c.old_hash);
      add_key(c.new_hash);
      unite_key(c.old_hash, c.new_hash);
    }

    std::map<uint64_t, std::pair<std::set<uint64_t>, std::set<uint64_t>>> raw;
    for (const Change& c : changes) {
      const uint64_t root = find_key(c.old_hash);
      raw[root].first.insert(c.old_hash);
      raw[root].second.insert(c.new_hash);
    }

    std::map<uint64_t, std::pair<std::set<uint64_t>, std::set<uint64_t>>> comps;
    for (const auto& r : raw) {
      uint64_t key = UINT64_MAX;
      for (uint64_t h : r.second.first)
        key = std::min(key, h);
      for (uint64_t h : r.second.second)
        key = std::min(key, h);
      comps[key] = r.second;
    }

    for (const auto& kv : comps) {
      std::map<std::string, int> reactant_mols;
      std::map<std::string, int> product_mols;
      std::map<std::string, int> reactant_elems;
      std::map<std::string, int> product_elems;
      int old_atoms = 0;
      int new_atoms = 0;

      for (uint64_t h : kv.second.first) {
        auto it = prev_registry_.find(h);
        if (it == prev_registry_.end())
          continue;
        old_atoms += it->second.atoms;
        reactant_mols[it->second.formula] += 1;
        for (const auto& e : it->second.elems)
          reactant_elems[e.first] += e.second;
      }
      for (uint64_t h : kv.second.second) {
        auto it = curr_registry_.find(h);
        if (it == curr_registry_.end())
          continue;
        new_atoms += it->second.atoms;
        product_mols[it->second.formula] += 1;
        for (const auto& e : it->second.elems)
          product_elems[e.first] += e.second;
      }

      const bool balanced = (reactant_elems == product_elems);
      const bool atom_conserved = (old_atoms == new_atoms);
      if (cfg_.require_balance && !balanced)
        continue;

      std::string rtext;
      for (const auto& e : reactant_mols) {
        for (int n = 0; n < e.second; ++n) {
          if (!rtext.empty())
            rtext += "+";
          rtext += e.first;
        }
      }
      std::string ptext;
      for (const auto& e : product_mols) {
        for (int n = 0; n < e.second; ++n) {
          if (!ptext.empty())
            ptext += "+";
          ptext += e.first;
        }
      }

      std::fprintf(
        fevents_,
        "%lld,%d,%d,%d,%s,%s,%d,%d\n",
        step,
        event_id_++,
        static_cast<int>(kv.second.first.size()),
        static_cast<int>(kv.second.second.size()),
        rtext.c_str(),
        ptext.c_str(),
        balanced ? 1 : 0,
        atom_conserved ? 1 : 0);
    }

    prev_registry_ = curr_registry_;
  }

  void write_state()
  {
    FILE* fp = std::fopen(cfg_.restart_file.c_str(), "wb");
    if (!fp) {
      std::fprintf(
        stderr,
        "compute_react_count: cannot write restart state '%s'\n",
        cfg_.restart_file.c_str());
      return;
    }
    const int magic = 0x52435431;
    const int version = 1;
    const int first = cfg_.offset_steps ? static_cast<int>(last_step_) : 0;
    const int natoms = n_;
    std::fwrite(&magic, sizeof(int), 1, fp);
    std::fwrite(&version, sizeof(int), 1, fp);
    std::fwrite(&event_id_, sizeof(int), 1, fp);
    std::fwrite(&first, sizeof(int), 1, fp);
    std::fwrite(&natoms, sizeof(int), 1, fp);
    for (int i = 0; i < n_; ++i) {
      const long long tag = i + 1;
      const unsigned long long hash = prev_hash_[i];
      std::fwrite(&tag, sizeof(long long), 1, fp);
      std::fwrite(&hash, sizeof(unsigned long long), 1, fp);
    }

    const int nreg = static_cast<int>(prev_registry_.size());
    std::fwrite(&nreg, sizeof(int), 1, fp);
    for (const auto& kv : prev_registry_) {
      const unsigned long long hash = kv.first;
      const int atoms = kv.second.atoms;
      const int flen = static_cast<int>(kv.second.formula.size());
      std::fwrite(&hash, sizeof(unsigned long long), 1, fp);
      std::fwrite(&atoms, sizeof(int), 1, fp);
      std::fwrite(&flen, sizeof(int), 1, fp);
      std::fwrite(kv.second.formula.data(), 1, flen, fp);
      const int nelem = static_cast<int>(kv.second.elems.size());
      std::fwrite(&nelem, sizeof(int), 1, fp);
      for (const auto& e : kv.second.elems) {
        const int slen = static_cast<int>(e.first.size());
        std::fwrite(&slen, sizeof(int), 1, fp);
        std::fwrite(e.first.data(), 1, slen, fp);
        std::fwrite(&e.second, sizeof(int), 1, fp);
      }
    }
    std::fclose(fp);
  }

  void read_state()
  {
    if (cfg_.restart_file.empty())
      return;
    FILE* fp = std::fopen(cfg_.restart_file.c_str(), "rb");
    if (!fp)
      return;

    int magic = 0, version = 0, event = 0, first = 0, natoms = 0;
    if (
      std::fread(&magic, sizeof(int), 1, fp) != 1 || magic != 0x52435431 ||
      std::fread(&version, sizeof(int), 1, fp) != 1 || version != 1 ||
      std::fread(&event, sizeof(int), 1, fp) != 1 || std::fread(&first, sizeof(int), 1, fp) != 1 ||
      std::fread(&natoms, sizeof(int), 1, fp) != 1) {
      std::fclose(fp);
      std::fprintf(
        stderr,
        "compute_react_count: ignoring malformed restart state '%s'\n",
        cfg_.restart_file.c_str());
      return;
    }
    if (cfg_.offset_steps)
      step_offset_ = first;
    if (natoms != n_) {
      std::fclose(fp);
      std::fprintf(
        stderr,
        "compute_react_count: restart state has %d atoms, current "
        "model has %d; ignoring\n",
        natoms,
        n_);
      return;
    }

    std::vector<int> tags(natoms, 0);
    std::vector<unsigned long long> hashes(natoms, 0);
    for (int i = 0; i < natoms; ++i) {
      long long tag = 0;
      unsigned long long hash = 0;
      if (
        std::fread(&tag, sizeof(long long), 1, fp) != 1 ||
        std::fread(&hash, sizeof(unsigned long long), 1, fp) != 1) {
        std::fclose(fp);
        std::fprintf(
          stderr,
          "compute_react_count: truncated restart state '%s'; ignoring\n",
          cfg_.restart_file.c_str());
        return;
      }
      tags[i] = static_cast<int>(tag);
      hashes[i] = hash;
    }

    std::vector<uint64_t> loaded(n_, 0);
    std::vector<char> seen(n_, 0);
    for (int i = 0; i < natoms; ++i) {
      const int index = tags[i] - 1;
      if (index < 0 || index >= n_)
        continue;
      loaded[index] = hashes[i];
      seen[index] = 1;
    }

    int nreg = 0;
    if (std::fread(&nreg, sizeof(int), 1, fp) != 1) {
      std::fclose(fp);
      return;
    }
    std::unordered_map<uint64_t, SpecInfo> registry;
    registry.reserve(nreg);
    for (int r = 0; r < nreg; ++r) {
      unsigned long long hash = 0;
      int atoms = 0, flen = 0;
      if (
        std::fread(&hash, sizeof(unsigned long long), 1, fp) != 1 ||
        std::fread(&atoms, sizeof(int), 1, fp) != 1 || std::fread(&flen, sizeof(int), 1, fp) != 1 ||
        flen < 0 || flen > 4096) {
        std::fclose(fp);
        return;
      }
      std::string formula(flen, '\0');
      if (flen > 0 && std::fread(formula.data(), 1, flen, fp) != static_cast<size_t>(flen)) {
        std::fclose(fp);
        return;
      }
      int nelem = 0;
      if (std::fread(&nelem, sizeof(int), 1, fp) != 1 || nelem < 0 || nelem > 128) {
        std::fclose(fp);
        return;
      }
      std::map<std::string, int> elems;
      for (int e = 0; e < nelem; ++e) {
        int slen = 0, count = 0;
        if (std::fread(&slen, sizeof(int), 1, fp) != 1 || slen < 0 || slen > 16) {
          std::fclose(fp);
          return;
        }
        std::string sym(slen, '\0');
        if (slen > 0 && std::fread(sym.data(), 1, slen, fp) != static_cast<size_t>(slen)) {
          std::fclose(fp);
          return;
        }
        if (std::fread(&count, sizeof(int), 1, fp) != 1) {
          std::fclose(fp);
          return;
        }
        elems[sym] = count;
      }
      registry[hash] = SpecInfo{formula, elems, atoms};
    }
    std::fclose(fp);

    int missing = 0;
    for (int i = 0; i < n_; ++i)
      if (!seen[i])
        ++missing;
    if (missing > 0) {
      std::fprintf(
        stderr,
        "compute_react_count: restart state does not cover %d "
        "atoms; event stream may be incomplete\n",
        missing);
    }

    prev_hash_ = std::move(loaded);
    prev_registry_ = std::move(registry);
    event_id_ = event;
    first_frame_ = false;
  }
};

} // namespace react_count

#ifndef REACT_COUNT_CORE_ONLY

class ComputeReactCount : public Action
{
public:
  explicit ComputeReactCount(const std::vector<std::string>& tokens)
  {
    parse(tokens);
    action_name = "compute_react_count";
  }

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override
  {
    (void)number_of_steps;
    (void)time_step;
    (void)integrate;
    (void)group;
    (void)box;
    (void)force;

    n_atoms_ = atom.number_of_atoms;
    const int ntypes = static_cast<int>(atom.cpu_type_size.size());
    if (ntypes <= 0)
      PRINT_INPUT_ERROR("compute_react_count found no atom types.");
    if (ntypes > react_count::MAXELEMS)
      PRINT_INPUT_ERROR("compute_react_count supports at most 32 atom types.");

    std::vector<std::string> symbols(ntypes);
    if (!element_list_.empty()) {
      if (static_cast<int>(element_list_.size()) != ntypes)
        PRINT_INPUT_ERROR("compute_react_count element list does not match the "
                          "number of types.");
      symbols = element_list_;
    } else {
      for (int i = 0; i < n_atoms_; ++i) {
        const int t = atom.cpu_type[i];
        if (t >= 0 && t < ntypes && symbols[t].empty())
          symbols[t] = atom.cpu_atom_symbol[i];
      }
      for (int t = 0; t < ntypes; ++t) {
        if (symbols[t].empty())
          symbols[t] = std::to_string(t);
      }
    }

    cfg_.make_elements(symbols, radius_override_, valence_override_);
    cfg_.offset_steps = true;
    engine_.reset(new react_count::Engine(cfg_));
    engine_->open(n_atoms_);

    std::printf("Compute transient-bond reaction statistics.\n");
    std::printf("    interval %d steps, output base %s\n", cfg_.nfreq, cfg_.output_base.c_str());
    std::printf(
      "    cutoff %.3f, max_neigh %d, perception %s\n",
      cfg_.cutoff,
      cfg_.max_neigh,
      cfg_.perception == 0 ? "reax" : "mutual");
    std::printf("    %d atoms, %d types", n_atoms_, ntypes);
    for (int t = 0; t < ntypes; ++t)
      std::printf(" %s(r=%.2f,v=%d)", symbols[t].c_str(), cfg_.radius[t], cfg_.valence[t]);
    std::printf("\n");
  }

  void end_of_step(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force) override
  {
    (void)number_of_steps;
    (void)fixed_group;
    (void)move_group;
    (void)global_time;
    (void)temperature;
    (void)integrate;
    (void)group;
    (void)thermo;
    (void)force;

    if ((step + 1) % cfg_.nfreq != 0)
      return;
    if (atom.number_of_atoms != n_atoms_)
      PRINT_INPUT_ERROR("compute_react_count atom count changed during the run.");

    std::vector<int> type(n_atoms_, 0);
    std::vector<double> position(static_cast<size_t>(n_atoms_) * 3, 0.0);
    std::vector<double> xyz(static_cast<size_t>(n_atoms_) * 3, 0.0);
    atom.type.copy_to_host(type.data());
    atom.position_per_atom.copy_to_host(position.data());
    // GPUMD stores positions as x[N], y[N], z[N]; the core uses xyzxyz.
    for (int i = 0; i < n_atoms_; ++i) {
      xyz[3 * i] = position[i];
      xyz[3 * i + 1] = position[n_atoms_ + i];
      xyz[3 * i + 2] = position[2 * n_atoms_ + i];
    }

    react_count::Box rb;
    rb.set_from_gpumd(box.cpu_h, box.pbc_x, box.pbc_y, box.pbc_z);
    engine_->step(step + 1, n_atoms_, type.data(), xyz.data(), rb);
  }

  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override
  {
    (void)atom;
    (void)box;
    (void)integrate;
    (void)number_of_steps;
    (void)time_step;
    (void)temperature;
    if (engine_)
      engine_->close();
    engine_.reset();
  }

private:
  react_count::Config cfg_;
  std::vector<std::string> element_list_;
  std::map<std::string, double> radius_override_;
  std::map<std::string, int> valence_override_;
  int n_atoms_ = 0;
  std::unique_ptr<react_count::Engine> engine_;

  static std::string token_value(const std::vector<std::string>& tokens, int& i)
  {
    if (i + 1 >= static_cast<int>(tokens.size()))
      PRINT_INPUT_ERROR("compute_react_count keyword is missing a value.");
    return tokens[++i];
  }

  static std::pair<std::string, std::string> split_pair(const std::string& text)
  {
    const size_t pos = text.find(':');
    if (pos == std::string::npos)
      PRINT_INPUT_ERROR("compute_react_count radius/valence value must be E:value.");
    return {text.substr(0, pos), text.substr(pos + 1)};
  }

  void parse(const std::vector<std::string>& tokens)
  {
    if (tokens.size() < 3)
      PRINT_INPUT_ERROR("compute_react_count should have 2 parameters: "
                        "interval output_base.");

    if (!is_valid_int(tokens[1], &cfg_.nfreq) || cfg_.nfreq <= 0)
      PRINT_INPUT_ERROR("compute_react_count interval should be a positive integer.");
    cfg_.output_base = tokens[2];

    for (int i = 3; i < static_cast<int>(tokens.size()); ++i) {
      const std::string& key = tokens[i];
      if (key == "cutoff") {
        if (!is_valid_real(token_value(tokens, i), &cfg_.cutoff) || cfg_.cutoff <= 0.0)
          PRINT_INPUT_ERROR("compute_react_count cutoff should be positive.");
      } else if (key == "element") {
        element_list_ = react_count::split_commas(token_value(tokens, i));
        if (element_list_.empty())
          PRINT_INPUT_ERROR("compute_react_count element list is empty.");
      } else if (key == "radius") {
        const auto pair = split_pair(token_value(tokens, i));
        double value = 0.0;
        if (!is_valid_real(pair.second, &value) || value < 0.0)
          PRINT_INPUT_ERROR("compute_react_count radius should be non-negative.");
        radius_override_[pair.first] = value;
      } else if (key == "valence") {
        const auto pair = split_pair(token_value(tokens, i));
        int value = 0;
        if (!is_valid_int(pair.second, &value) || value < 0)
          PRINT_INPUT_ERROR("compute_react_count valence should be non-negative.");
        valence_override_[pair.first] = value;
      } else if (key == "max_neigh") {
        if (!is_valid_int(token_value(tokens, i), &cfg_.max_neigh) || cfg_.max_neigh <= 0)
          PRINT_INPUT_ERROR("compute_react_count max_neigh should be positive.");
      } else if (key == "order") {
        cfg_.order = react_count::split_commas(token_value(tokens, i));
        if (cfg_.order.empty())
          PRINT_INPUT_ERROR("compute_react_count order list is empty.");
      } else if (key == "events") {
        if (!react_count::parse_bool(token_value(tokens, i), cfg_.events))
          PRINT_INPUT_ERROR("compute_react_count events should be yes/no.");
      } else if (key == "require_balance") {
        if (!react_count::parse_bool(token_value(tokens, i), cfg_.require_balance))
          PRINT_INPUT_ERROR("compute_react_count require_balance should be yes/no.");
        if (cfg_.require_balance)
          cfg_.events = true;
      } else if (key == "restart") {
        cfg_.restart_file = token_value(tokens, i);
      } else if (key == "perception") {
        const std::string value = token_value(tokens, i);
        if (value == "reax")
          cfg_.perception = 0;
        else if (value == "mutual")
          cfg_.perception = 1;
        else
          PRINT_INPUT_ERROR("compute_react_count perception should be reax or mutual.");
      } else {
        PRINT_INPUT_ERROR("compute_react_count has an unknown keyword.");
      }
    }
  }
};

std::unique_ptr<Action> create_compute_react_count(const std::vector<std::string>& tokens)
{
  return std::unique_ptr<Action>(new ComputeReactCount(tokens));
}

#endif
