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

#pragma once

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

class Parameters;

struct Neighbor {
  int i;
  int j;
  int sx;
  int sy;
  int sz;
  double dx;
  double dy;
  double dz;
  double r;
};

class Dataset
{
public:
  struct Atom {
    std::string symbol;
    int type = -1;
    std::array<double, 3> position;
  };

  struct Structure {
    int num_atoms = 0;
    std::array<std::array<double, 3>, 3> lattice{};
    std::vector<Atom> atoms;
    double volume = 0.0;
  };

  int index = -1;

  Structure structure;
  std::vector<std::array<double, 3>> kpoints;
  std::vector<std::array<double, 3>> kpoints_cart;
  std::vector<std::vector<double>> bands;

  std::vector<Neighbor> neighbors_radial;
  std::vector<Neighbor> neighbors_angular;
  std::vector<Neighbor> hopping_bonds;

private:
  friend class DatasetCollection;
  Dataset() = default;
  void build_cartesian_kpoints();
  void build_neighbors(const Parameters& para);
};

class DatasetCollection
{
public:
  DatasetCollection(const Parameters& para, const std::string& filename, bool optional = false);

  int size() const { return static_cast<int>(structures_.size()); }
  const Dataset& operator[](int i) const { return structures_[i]; }
  void print_summary() const;
  bool has_reference() const { return !structures_.empty() && !structures_.front().bands.empty(); }

  class Batch
  {
  public:
    Batch(const Parameters& para, const DatasetCollection& dataset);
    void next();
    int estimated_steps_per_epoch() const;
    const std::vector<int>& structure_indices() const { return structure_indices_; }
    const std::vector<int>& k_indices(int s) const { return k_indices_[s]; }

  private:
    std::vector<int> structure_permutation_;
    std::size_t structure_cursor_ = 0;
    std::mt19937 structure_rng_;
    std::vector<int> structure_indices_;
    std::vector<std::vector<int>> k_permutations_;
    std::vector<std::size_t> k_cursors_;
    std::vector<std::mt19937> k_rngs_;
    std::vector<std::vector<int>> k_indices_;
  };

private:
  std::vector<Dataset> structures_;
};
