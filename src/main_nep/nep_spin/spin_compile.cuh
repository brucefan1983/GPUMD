/* This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "force/nep_spin/spin_layout.cuh"
#include <iomanip>
#include <sstream>

namespace nep_spin {
// Bake the existing layout tables into getters, so unrolled accesses are constants.
// This generates metadata only; all forward and adjoint arithmetic stays in the shared kernel.
inline std::string compile_layout(const Layout& v, int soc, const char* table = "aggregate_terms") {
  std::ostringstream out;
  out << std::scientific << std::setprecision(17);
  auto term = [&](const Term& p) {
    out << '{' << p.geometry << ',' << p.spin << ',' << p.output << ','
        << p.path << ',' << p.neighbor << ',' << p.coefficient << "f}";
  };
  out << "namespace nep_spin { ";
  // Warp lanes read the same term. Tables above 64 KiB use read-only global memory.
  out << "__device__ " << (v.terms.size()*sizeof(Term) <= 64*1024 ? "__constant__" : "const")
      << " Term " << table << "[] = {\n";
  for (const auto& p : v.terms) {
    term(p);
    out << ",\n";
  }
  out << "};\n";
  out << "struct CompiledLayout {\n";
  auto count = [&](const char* name, int value) {
    out << "enum : int { " << name << " = " << value << " };\n";
  };
  count("edge_cached", v.lmax == 2 && v.order == 2 ? 1 : 0);
  out << "const float* edge_radial = nullptr;\n";
  count("tile", v.lmax == 2 && v.order == 2 ? 2 : 1);
  count("channels", v.channels); count("lmax", v.lmax);
  count("order", v.order); count("soc", soc);
  count("primitive_count", v.primitives.size()); count("sector_count", v.sectors.size());
  count("term_count", v.terms.size()); count("center_count", v.centers.size());
  count("path_count", v.path_count); count("components", v.components);
  count("component_capacity", v.components); count("spin_capacity", v.spin_components);
  count("spin_components", v.spin_components);
  count("harmonic_capacity", (v.lmax+1)*(v.lmax+1));
  auto start = [&](const char* type, const char* name) {
    out << "__device__ __forceinline__ " << type << " " << name
        << "(int i) const { switch(i) {\n";
  };
  auto end = [&]() { out << "default: return {}; } }\n"; };
  start("Primitive", "primitive");
  for (int i = 0; i < v.primitives.size(); ++i) {
    const auto& p = v.primitives[i];
    out << "case " << i << ": return {" << p.degree << ',' << p.l << ',' << p.offset << "};\n";
  }
  end(); start("Sector", "sector");
  for (int i = 0; i < v.sectors.size(); ++i) {
    const auto& p = v.sectors[i];
    out << "case " << i << ": return {" << p.l << ',' << p.parity << ',' << p.time << ','
        << p.offset << ',' << p.norm_column << ',' << p.scalar_column << ',' << p.size << "};\n";
  }
  end();
  out << "__device__ Term term(int i) const { return " << table << "[i]; }\n";
  start("Center", "center");
  for (int i = 0; i < v.centers.size(); ++i) {
    const auto& p = v.centers[i];
    out << "case " << i << ": return {" << p.primitive << ',' << p.output << ',' << p.weight << "};\n";
  }
  end(); start("Harmonic", "harmonic");
  int n = 0;
  for (const auto& p : harmonics) {
    if (p.component >= (v.lmax+1)*(v.lmax+1)) continue;
    out << "case " << n++ << ": return {" << p.component << ',' << p.x << ',' << p.y << ','
        << p.z << ',' << p.value << "};\n";
  }
  end(); count("harmonic_count", n);
  out << "}; }\n";
  return out.str();
}
} // namespace nep_spin
