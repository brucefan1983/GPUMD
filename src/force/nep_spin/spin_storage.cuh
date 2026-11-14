/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "spin_layout.cuh"
#include "utilities/gpu_vector.cuh"

namespace nep_spin {
struct Storage {
  GPU_Vector<Primitive> primitives;
  GPU_Vector<Sector> sectors;
  GPU_Vector<Term> terms;
  GPU_Vector<Center> centers;
  GPU_Vector<Harmonic> harmonic_data;
  View view{};
  void initialize(int c, int l, int s, int soc) {
    Layout layout(c, l, s, soc);
    primitives.resize(layout.primitives.size());
    primitives.copy_from_host(layout.primitives.data());
    sectors.resize(layout.sectors.size());
    sectors.copy_from_host(layout.sectors.data());
    terms.resize(layout.terms.size());
    terms.copy_from_host(layout.terms.data());
    centers.resize(layout.centers.size());
    if (!layout.centers.empty()) centers.copy_from_host(layout.centers.data());
    const int count = sizeof(harmonics) / sizeof(harmonics[0]);
    harmonic_data.resize(count);
    harmonic_data.copy_from_host(harmonics);
    view = {c, s, l, static_cast<int>(layout.primitives.size()), layout.spin_components,
            layout.components, static_cast<int>(layout.sectors.size()),
            static_cast<int>(layout.terms.size()), static_cast<int>(layout.centers.size()),
            layout.path_count, layout.local_count, layout.dimension,
            primitives.data(), sectors.data(), terms.data(), centers.data(), harmonic_data.data(), count};
  }
};
} // namespace nep_spin
