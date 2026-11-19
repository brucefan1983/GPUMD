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

// Chengbing Chen: GPU storage and indexing for packed spin projections.
#pragma once
#include "spin_layout.cuh"
#include "utilities/gpu_vector.cuh"

namespace nep_spin
{
struct ProjectionView {
  const ProjectionRow* rows;
  const ProjectionTerm* terms;
  const float* neighbor_weights;
  const int* paths;

  template <bool CandidateWeights>
  __device__ __forceinline__ static int term_index(int begin, int j, int c, int channels)
  {
    return begin + (CandidateWeights ? j : j * channels + c) * SpinProjection::width;
  }

  template <bool CandidateWeights>
  __device__ __forceinline__ ProjectionTerm
  term(ProjectionRow row, int j, int c, int channels, const float* weights) const
  {
    const int index = term_index<CandidateWeights>(row.begin, j, c, channels);
    ProjectionTerm entry = terms[index];
    if (CandidateWeights)
      entry.weight *= weights[paths[index] * channels + c];
    return entry;
  }

  // Paired rows share one adjoint input between the center and neighbor couplings.
  template <bool CandidateWeights, class Consume>
  __device__ __forceinline__ int
  for_each_pair(int m, int c, int channels, const float* weights, Consume consume) const
  {
    auto visit = [&](int index) {
      const ProjectionTerm entry = terms[index];
      float center = entry.weight;
      float neighbor = neighbor_weights[index];
      if (CandidateWeights) {
        center *= weights[paths[2 * index] * channels + c];
        neighbor *= weights[paths[2 * index + 1] * channels + c];
      }
      consume(entry.input, center, neighbor);
    };
    if (rows) {
      const ProjectionRow row = rows[m];
      for (int j = 0; j < row.count; ++j)
        visit(term_index<CandidateWeights>(row.begin, j, c, channels));
      return row.output;
    }
    constexpr int width = SpinProjection::width;
    visit(CandidateWeights ? m : (m / width * channels + c) * width + m % width);
    return m;
  }
};
struct ProjectionStorage {
  GPU_Vector<ProjectionRow> rows;
  GPU_Vector<ProjectionTerm> terms;
  GPU_Vector<float> neighbor_weights;
  GPU_Vector<int> paths;
  bool has_pairs = false;
  bool single_term_rows = false;

  void initialize(const Layout& layout, const float* weights, bool transpose, bool paired = false)
  {
    SpinProjection projection(layout, weights, transpose, paired);
    has_pairs = paired;
    single_term_rows =
      paired &&
      std::all_of(projection.rows.begin(), projection.rows.end(), [](const ProjectionRow& row) {
        return row.count == 1;
      });
    if (!single_term_rows) {
      rows.resize(projection.rows.size());
      rows.copy_from_host(projection.rows.data());
    }
    terms.resize(projection.packed.size());
    terms.copy_from_host(projection.packed.data());
    if (!projection.paths.empty()) {
      paths.resize(projection.paths.size());
      paths.copy_from_host(projection.paths.data());
    }
    if (paired) {
      neighbor_weights.resize(projection.neighbor_weights.size());
      neighbor_weights.copy_from_host(projection.neighbor_weights.data());
    }
  }

  bool paired() const { return has_pairs; }

  ProjectionView view() const
  {
    // Equally sized one-term rows stay in their original order and need no index table.
    return {
      single_term_rows ? nullptr : rows.data(),
      terms.data(),
      has_pairs ? neighbor_weights.data() : nullptr,
      paths.size() ? paths.data() : nullptr};
  }
};

struct Storage {
  GPU_Vector<Primitive> primitives;
  GPU_Vector<Sector> sectors;
  GPU_Vector<Term> terms;
  GPU_Vector<int> adjoint_order;
  ProjectionStorage forward, transpose;
  GPU_Vector<Center> centers;
  GPU_Vector<Harmonic> harmonic_data;
  View view{};
  void initialize(
    int c,
    int l,
    int s,
    int soc,
    int adjoint_lanes = 1,
    const float* fixed_weights = nullptr,
    bool project_forward = false,
    bool paired = false)
  {
    Layout layout(c, l, s, soc);
    const int elements = (l + 1) * (l + 1) * layout.spin_components;
    if (
      fixed_weights && compact_fused_rows(adjoint_lanes, elements, c > 1) &&
      layout.terms.size() <= size_t(2 * elements))
      paired = false;
    primitives.resize(layout.primitives.size());
    primitives.copy_from_host(layout.primitives.data());
    sectors.resize(layout.sectors.size());
    sectors.copy_from_host(layout.sectors.data());
    terms.resize(layout.terms.size());
    terms.copy_from_host(layout.terms.data());
    if (!paired && adjoint_lanes > 1 && adjoint_lanes != SpinProjection::width) {
      // Offsets followed by term indices. Each geometry row has one writer, in model order.
      std::vector<int> order(adjoint_lanes + 1);
      for (int lane = 0; lane < adjoint_lanes; ++lane) {
        order[lane] = static_cast<int>(order.size());
        for (int t = 0; t < static_cast<int>(layout.terms.size()); ++t)
          if (layout.terms[t].geometry % adjoint_lanes == lane)
            order.push_back(t);
      }
      order[adjoint_lanes] = static_cast<int>(order.size());
      adjoint_order.resize(order.size());
      adjoint_order.copy_from_host(order.data());
    }
    if (project_forward)
      forward.initialize(layout, fixed_weights, false);
    if (paired || adjoint_lanes == SpinProjection::width)
      transpose.initialize(layout, fixed_weights, true, paired);
    centers.resize(layout.centers.size());
    if (!layout.centers.empty())
      centers.copy_from_host(layout.centers.data());
    const int count = sizeof(harmonics) / sizeof(harmonics[0]);
    harmonic_data.resize(count);
    harmonic_data.copy_from_host(harmonics);
    view = {
      c,
      s,
      l,
      static_cast<int>(layout.primitives.size()),
      layout.spin_components,
      layout.components,
      static_cast<int>(layout.sectors.size()),
      static_cast<int>(layout.terms.size()),
      static_cast<int>(layout.centers.size()),
      layout.path_count,
      layout.local_count,
      layout.dimension,
      primitives.data(),
      sectors.data(),
      terms.data(),
      centers.data(),
      harmonic_data.data(),
      count};
  }
};
} // namespace nep_spin
