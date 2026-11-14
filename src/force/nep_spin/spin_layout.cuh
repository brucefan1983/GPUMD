/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "spin_tables.cuh"
#include <algorithm>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace nep_spin {
inline bool needs_wide_forward(int soc, int lmax, int order) {
  return !soc && lmax == 4 && order == 4;
}
struct Primitive { int degree, l, offset; };
struct Sector { int l, parity, time, offset, norm_column, scalar_column, size; };
struct Term { int geometry, spin, output, path, neighbor; float coefficient; };
struct Center { int primitive, output, weight; };
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
  Layout(int c, int l, int s, int soc = 1) : channels(c), order(s), lmax(l) {
    for (int d = 0; d <= s; ++d)
      for (int j = d % 2; j <= std::min(d, l); j += 2) {
        primitives.push_back({d, j, spin_components});
        spin_components += 2 * j + 1;
        if (d && j == 0) ++local_count;
      }
    if (!soc) {
      // Keep the spatial and spin irreps separate: each sector transforms as
      // D^g(R_space) x D^j(R_spin). Only its full Frobenius norm is invariant.
      // Degrees with the same spin rank can mix, as can center/neighbor spins.
      dimension = local_count;
      for (int g = 0; g <= l; ++g) {
        for (int j = 0; j <= std::min(l, s); ++j) {
          const int width = (2*g+1)*(2*j+1);
          int scalar = -1;
          if (g == 0 && j == 0) { scalar = dimension; dimension += c; }
          const int offset = components;
          sectors.push_back({j, g%2 ? -1 : 1, j%2 ? -1 : 1,
                             offset, dimension, scalar, width});
          dimension += c;
          components += width;
          for (int p = 0; p < static_cast<int>(primitives.size()); ++p) {
            const auto& primitive = primitives[p];
            if (primitive.l != j) continue;
            for (int site = 0; site < (primitive.degree ? 2 : 1); ++site) {
              for (int m = 0; m < 2*g+1; ++m)
                for (int a = 0; a < 2*j+1; ++a)
                  terms.push_back({g*g+m, primitive.offset+a,
                    offset+m*(2*j+1)+a, path_count, site, 1.0f});
              ++path_count;
            }
            // An on-site spin tensor has spatial rank zero only.
            if (g == 0 && p > 0)
              for (int a = 0; a < 2*j+1; ++a)
                centers.push_back({primitive.offset+a, offset+a, p-1});
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
      if (key == Key(0, 1, 1)) { scalar = dimension; dimension += c; }
      sectors.push_back({j, parity, time, components, dimension, scalar, 2*j+1});
      dimension += c;
      components += 2 * j + 1;
    }
    for (int g = 0; g <= l; ++g)
      for (const auto& p : primitives)
        for (int site = 0; site < (p.degree ? 2 : 1); ++site)
          for (int o = std::abs(g - p.l); o <= std::min(g + p.l, l); ++o) {
            const Key key(o, g % 2 ? -1 : 1, p.degree % 2 ? -1 : 1);
            const auto& sector = sectors[std::lower_bound(keys.begin(), keys.end(), key) - keys.begin()];
            for (const auto& cg : couplings)
              if (cg.l1 == g && cg.l2 == p.l && cg.lout == o)
                terms.push_back({g*g+cg.a, p.offset+cg.b, sector.offset+cg.m,
                                 path_count, site, static_cast<float>(cg.value)});
            ++path_count;
          }
    for (const auto& sector : sectors)
      for (int p = 1; p < static_cast<int>(primitives.size()); ++p) {
        const auto& primitive = primitives[p];
        if (sector.l == primitive.l && sector.parity == 1 &&
            sector.time == (primitive.degree % 2 ? -1 : 1))
          for (int m = 0; m < 2 * primitive.l + 1; ++m)
            centers.push_back({primitive.offset+m, sector.offset+m, p-1});
      }
  }
  int parameter_count(int radial_channels) const {
    return channels * (radial_channels + path_count + static_cast<int>(primitives.size()) - 1);
  }
};
} // namespace nep_spin
