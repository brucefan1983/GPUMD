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

// Chengbing Chen: packed forward and transposed spin projections.
#pragma once
#include "spin_tables.cuh"
#include <algorithm>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace nep_spin
{
constexpr bool needs_wide_forward(int soc, int lmax, int order)
{
  return !soc && lmax == 4 && order == 4;
}
template <class Real>
struct alignas(2 * sizeof(Real)) RadialEdge {
  Real value, derivative;
};

// Each atom/channel owns a tile of consecutive neighbors. Adjacent work groups own adjacent tiles.
template <int Tile>
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
  constexpr size_t
  radial_edge_index(int atoms, int channels, int atom, int channel, int neighbor)
{
  return ((size_t(neighbor / Tile) * atoms + atom) * channels + channel) * Tile + neighbor % Tile;
}

struct Primitive {
  int degree, l, offset;
};
// All C and SOC choices with the same L and order use the same spin basis.
struct PrimitiveBasis {
  Primitive entries[9] = {};
  int count = 0, components = 0, scalars = 0;
#if defined(__CUDACC__) || defined(__HIPCC__)
  __host__ __device__
#endif
    constexpr PrimitiveBasis(int lmax, int order)
  {
    for (int d = 0; d <= order; ++d)
      for (int j = d % 2; j <= d && j <= lmax; j += 2) {
        entries[count++] = {d, j, components};
        components += 2 * j + 1;
        if (d && j == 0)
          ++scalars;
      }
  }
};
struct Sector {
  int l, parity, time, offset, norm_column, scalar_column, size;
};
struct Term {
  int geometry, spin, output, path, neighbor;
  float coefficient;
};
struct Center {
  int primitive, output, weight;
};
struct View {
  int channels, order, lmax, primitive_count, spin_components, components;
  int sector_count, term_count, center_count, path_count, local_count, dimension;
  const Primitive* primitives;
  const Sector* sectors;
  const Term* terms;
  const Center* centers;
  const Harmonic* harmonics;
  int harmonic_count;
};
struct Layout {
  std::vector<Primitive> primitives;
  std::vector<Sector> sectors;
  std::vector<Term> terms;
  std::vector<Center> centers;
  int channels, order, lmax, spin_components = 0, components = 0;
  int path_count = 0, local_count = 0, dimension = 0;
  Layout(int c, int l, int s, int soc = 1) : channels(c), order(s), lmax(l)
  {
    const PrimitiveBasis basis(l, s);
    primitives.assign(basis.entries, basis.entries + basis.count);
    spin_components = basis.components;
    local_count = basis.scalars;
    if (!soc) {
      // Keep the spatial and spin irreps separate: each sector transforms as
      // D^g(R_space) x D^j(R_spin). Only its full Frobenius norm is invariant.
      // Degrees with the same spin rank can mix, as can center/neighbor spins.
      dimension = local_count;
      for (int g = 0; g <= l; ++g) {
        for (int j = 0; j <= std::min(l, s); ++j) {
          const int width = (2 * g + 1) * (2 * j + 1);
          int scalar = -1;
          if (g == 0 && j == 0) {
            scalar = dimension;
            dimension += c;
          }
          const int offset = components;
          sectors.push_back({j, g % 2 ? -1 : 1, j % 2 ? -1 : 1, offset, dimension, scalar, width});
          dimension += c;
          components += width;
          for (int p = 0; p < static_cast<int>(primitives.size()); ++p) {
            const auto& primitive = primitives[p];
            if (primitive.l != j)
              continue;
            for (int site = 0; site < (primitive.degree ? 2 : 1); ++site) {
              for (int m = 0; m < 2 * g + 1; ++m)
                for (int a = 0; a < 2 * j + 1; ++a)
                  terms.push_back(
                    {g * g + m,
                     primitive.offset + a,
                     offset + m * (2 * j + 1) + a,
                     path_count,
                     site,
                     1.0f});
              ++path_count;
            }
            // An on-site spin tensor has spatial rank zero only.
            if (g == 0 && p > 0)
              for (int a = 0; a < 2 * j + 1; ++a)
                centers.push_back({primitive.offset + a, offset + a, p - 1});
          }
        }
      }
      return;
    }
    using Key = std::tuple<int, int, int>;
    std::vector<Key> keys;
    for (int g = 0; g <= l; ++g)
      for (const auto& p : primitives)
        for (int o = std::abs(g - p.l); o <= std::min(g + p.l, l); ++o)
          keys.emplace_back(o, g % 2 ? -1 : 1, p.degree % 2 ? -1 : 1);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    dimension = local_count;
    for (const auto& key : keys) {
      int j, parity, time;
      std::tie(j, parity, time) = key;
      int scalar = -1;
      if (key == Key(0, 1, 1)) {
        scalar = dimension;
        dimension += c;
      }
      sectors.push_back({j, parity, time, components, dimension, scalar, 2 * j + 1});
      dimension += c;
      components += 2 * j + 1;
    }
    for (int g = 0; g <= l; ++g)
      for (const auto& p : primitives)
        for (int site = 0; site < (p.degree ? 2 : 1); ++site)
          for (int o = std::abs(g - p.l); o <= std::min(g + p.l, l); ++o) {
            const Key key(o, g % 2 ? -1 : 1, p.degree % 2 ? -1 : 1);
            const auto& sector =
              sectors[std::lower_bound(keys.begin(), keys.end(), key) - keys.begin()];
            for (const auto& cg : couplings)
              if (cg.l1 == g && cg.l2 == p.l && cg.lout == o)
                terms.push_back(
                  {g * g + cg.a,
                   p.offset + cg.b,
                   sector.offset + cg.m,
                   path_count,
                   site,
                   static_cast<float>(cg.value)});
            ++path_count;
          }
    for (const auto& sector : sectors)
      for (int p = 1; p < static_cast<int>(primitives.size()); ++p) {
        const auto& primitive = primitives[p];
        if (
          sector.l == primitive.l && sector.parity == 1 &&
          sector.time == (primitive.degree % 2 ? -1 : 1))
          for (int m = 0; m < 2 * primitive.l + 1; ++m)
            centers.push_back({primitive.offset + m, sector.offset + m, p - 1});
      }
  }
  int parameter_count(int radial_channels) const
  {
    return channels * (radial_channels + path_count + static_cast<int>(primitives.size()) - 1);
  }
};

constexpr bool large_polynomial_density(int l, int order)
{
  return order > 2 && (l + 1) * (l + 1) * PrimitiveBasis(l, order).components >= 256;
}

constexpr bool warp_training_forward(int l, int order, int soc)
{
  return large_polynomial_density(l, order) && !needs_wide_forward(soc, l, order);
}

constexpr bool fused_training_adjoint(int c, int l, int order, int soc)
{
  return c > 1 && warp_training_forward(l, order, soc);
}

struct ProjectionRow {
  int output, begin, count;
};
struct alignas(8) ProjectionTerm {
  unsigned short input;
  short center_spin;
  float weight;
};

// Fused warps with at most ten density entries per lane can consume sparse rows directly.
constexpr bool compact_fused_rows(int lanes, int elements, bool fused)
{
  return lanes == 32 && fused && elements <= 10 * 32;
}

struct SpinProjection {
  enum { width = 32 };
  std::vector<ProjectionRow> rows;
  std::vector<ProjectionTerm> packed;
  std::vector<float> neighbor_weights;
  std::vector<int> paths; // One candidate-weight index per term, or a center/neighbor pair.
  SpinProjection(const Layout& layout, const float* weights, bool transpose, bool paired = false)
  {
    const int harmonics = (layout.lmax + 1) * (layout.lmax + 1);
    const int outputs = transpose ? harmonics * layout.spin_components : layout.components;
    auto density_index = [&](const Term& t) {
      return t.geometry * layout.spin_components + (t.neighbor ? t.spin : 0);
    };
    std::vector<std::vector<int>> by_output(outputs);
    for (int t = 0; t < static_cast<int>(layout.terms.size()); ++t) {
      const Term& term = layout.terms[t];
      const int density =
        paired ? term.geometry * layout.spin_components + term.spin : density_index(term);
      by_output[transpose ? density : term.output].push_back(t);
    }
    // Center and neighbor couplings for a given (geometry, spin, output) use the same moment.
    std::vector<int> partner(paired ? layout.terms.size() : 0, -1);
    if (paired) {
      for (auto& row : by_output) {
        std::vector<int> centers;
        for (int t : row)
          if (!layout.terms[t].neighbor)
            centers.push_back(t);
        for (int t : row) {
          if (!layout.terms[t].neighbor)
            continue;
          auto match = std::find_if(centers.begin(), centers.end(), [&](int local) {
            return partner[local] < 0 && layout.terms[local].output == layout.terms[t].output;
          });
          if (match == centers.end()) {
            centers.push_back(t);
            partner[t] = t;
          } else {
            partner[*match] = t;
          }
        }
        row.swap(centers);
      }
    }
    for (int m = 0; m < outputs; ++m)
      rows.push_back({m, 0, static_cast<int>(by_output[m].size())});
    // Group similar-length rows so lanes read adjacent entries and finish together.
    std::stable_sort(
      rows.begin(), rows.end(), [](ProjectionRow a, ProjectionRow b) { return a.count > b.count; });
    const int channels = weights ? layout.channels : 1;
    for (int base = 0; base < outputs; base += width) {
      const int start = static_cast<int>(packed.size());
      packed.resize(start + width * rows[base].count * channels);
      if (!weights)
        paths.resize(packed.size() * (paired ? 2 : 1));
      if (paired)
        neighbor_weights.resize(packed.size());
      for (int lane = 0; lane < width && base + lane < outputs; ++lane) {
        auto& row = rows[base + lane];
        row.begin = start + lane;
        for (int k = 0; k < row.count; ++k) {
          const int t = by_output[row.output][k];
          const Term& term = layout.terms[t];
          const int input = transpose ? term.output : density_index(term);
          for (int c = 0; c < channels; ++c) {
            const int offset = row.begin + (k * channels + c) * width;
            float weight = term.coefficient;
            if (weights)
              weight *= weights[term.path * channels + c];
            else
              paths[offset * (paired ? 2 : 1)] = term.path;
            packed[offset] = {
              static_cast<unsigned short>(input),
              static_cast<short>(term.neighbor ? -1 : term.spin),
              (paired && term.neighbor) ? 0.0f : weight};
            if (paired && partner[t] >= 0) {
              const auto& neighbor = layout.terms[partner[t]];
              neighbor_weights[offset] = neighbor.coefficient;
              if (weights)
                neighbor_weights[offset] *= weights[neighbor.path * channels + c];
              else
                paths[2 * offset + 1] = neighbor.path;
            }
          }
        }
      }
    }
  }
};
} // namespace nep_spin
