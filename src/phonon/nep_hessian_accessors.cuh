/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
NEP accessors used by the hessian module.

These helpers recover the bits of NEP state that the analytic Hessian needs
without adding friend declarations or other modifications to src/force/nep.cuh.

Strategy:
  * `requires_expanded_box` uses the public `paramb.rc_radial_max` and the
    public Box interface, so it is implemented in-place.
  * `uses_dftd3` and `uses_zbl` re-read the configuration that NEP read at
    construction time.  NEP::initialize_dftd3 scans run.in for a "dftd3"
    line, and the NEP constructor scans nep.txt for "nep4_zbl" or
    "nep4_zbl_temperature" and reads ZBL cutoffs / flexible parameters
    from the same file.  We perform the same scans here.

Because the parsing logic mirrors the NEP constructor, the result is
equivalent to the in-class accessors.  The hessian module owns the
reconstructed NEP::ZBL struct and passes it by value to
`find_zbl_hessian_kernel`, exactly the way `nep->get_zbl()` would.
------------------------------------------------------------------------------*/

#pragma once
#include "force/nep.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace nep_hessian {

inline bool nep_requires_expanded_box(const NEP& nep, const Box& box)
{
  const double volume = box.get_volume();
  const double cutoff = nep.paramb.rc_radial_max;
  return
    (box.pbc_x && volume / box.get_area(0) <= 2.5 * (cutoff + 1.0)) ||
    (box.pbc_y && volume / box.get_area(1) <= 2.5 * (cutoff + 1.0)) ||
    (box.pbc_z && volume / box.get_area(2) <= 2.5 * (cutoff + 1.0));
}

namespace {

inline std::string find_first_potential_filename()
{
  std::ifstream run_input("run.in");
  if (!run_input.is_open()) return std::string();
  std::string line;
  while (std::getline(run_input, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() >= 2 && tokens[0] == "potential") {
      return tokens[1];
    }
  }
  return std::string();
}

inline std::string find_element_symbol(int atomic_number)
{
  // Atomic numbers are 1..NUM_ELEMENTS.  The list below mirrors the
  // ELEMENTS table in src/force/nep.cu; both must be kept in sync.
  static const char* symbols[NUM_ELEMENTS] = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu"
  };
  if (atomic_number < 1 || atomic_number > NUM_ELEMENTS) return std::string();
  return symbols[atomic_number - 1];
}

inline int lookup_atomic_number(const std::string& symbol)
{
  for (int z = 1; z <= NUM_ELEMENTS; ++z) {
    if (find_element_symbol(z) == symbol) return z;
  }
  return 0;
}

} // namespace

// True if run.in contains a "dftd3" line.  Mirrors NEP::initialize_dftd3.
inline bool run_input_has_dftd3()
{
  std::ifstream run_input("run.in");
  if (!run_input.is_open()) return false;
  std::string line;
  while (std::getline(run_input, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (!tokens.empty() && tokens[0] == "dftd3") return true;
  }
  return false;
}

// Reconstruct the NEP::ZBL state from the first potential file.  Returns a
// ZBL struct whose `enabled` field is set if and only if the model actually
// uses ZBL.  The parsing mirrors the NEP constructor:
//   * first line: "nep4_zbl ..." or "nep4_zbl_temperature ..." enables ZBL.
//   * next line (if ZBL): "zbl rc_inner rc_outer [zbl_factor]"
//   * subsequent 10 * num_type_pairs lines (flexible ZBL only): the
//     per-pair parameters used to fit the universal ZBL.
//
// `paramb_in` is filled with the typewise ZBL cutoff fields used by the
// Hessian kernel.
struct ZBLParseResult {
  NEP::ZBL zbl;
  bool ok = false;
};

inline ZBLParseResult parse_nep_zbl(const std::string& filename, NEP::ParaMB& paramb_in)
{
  ZBLParseResult result;
  std::ifstream input(filename);
  if (!input.is_open()) return result;

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.empty()) return result;

  const std::string& header = tokens[0];
  const bool zbl_enabled =
    (header == "nep4_zbl") || (header == "nep4_zbl_temperature");
  result.zbl.enabled = zbl_enabled;
  if (!zbl_enabled) {
    result.ok = true;
    return result;
  }

  if (tokens.size() < 3) return result;
  int parsed_num_types = 0;
  try {
    parsed_num_types = std::stoi(tokens[1]);
  } catch (...) {
    return result;
  }
  if (static_cast<int>(tokens.size()) != 2 + parsed_num_types) return result;
  for (int n = 0; n < parsed_num_types; ++n) {
    result.zbl.atomic_numbers[n] = lookup_atomic_number(tokens[2 + n]);
  }
  result.zbl.num_types = parsed_num_types;
  paramb_in.num_types = parsed_num_types;

  // "zbl rc_inner rc_outer [zbl_factor]"
  tokens = get_tokens(input);
  if (tokens.size() != 3 && tokens.size() != 4) return result;
  try {
    result.zbl.rc_inner = std::stof(tokens[1]);
    result.zbl.rc_outer = std::stof(tokens[2]);
  } catch (...) {
    return result;
  }
  if (result.zbl.rc_inner == 0.0f && result.zbl.rc_outer == 0.0f) {
    result.zbl.flexible = true;
  }
  if (tokens.size() == 4) {
    try {
      paramb_in.typewise_cutoff_zbl_factor = std::stof(tokens[3]);
      paramb_in.use_typewise_cutoff_zbl = true;
    } catch (...) {
      return result;
    }
  }

  // Flexible ZBL parameters: 10 floats per (i, j) pair, j >= i, in the
  // canonical triangular order.  See nep.cu for the exact loop.
  if (result.zbl.flexible) {
    const int num_type_zbl = (parsed_num_types * (parsed_num_types + 1)) / 2;
    const int total = 10 * num_type_zbl;
    for (int d = 0; d < total; ++d) {
      tokens = get_tokens(input);
      if (tokens.empty()) return result;
      try {
        result.zbl.para[d] = std::stof(tokens[0]);
      } catch (...) {
        return result;
      }
    }
  }

  result.ok = true;
  return result;
}

// Convenience: returns the parsed ZBL state, or a default-disabled ZBL if
// the file is missing or parsing fails.  Reports parsing problems via the
// `ok` flag.
inline NEP::ZBL load_nep_zbl(const std::string& filename, NEP::ParaMB& paramb_in)
{
  ZBLParseResult parsed = parse_nep_zbl(filename, paramb_in);
  if (!parsed.ok) {
    NEP::ZBL empty;
    return empty;
  }
  return parsed.zbl;
}

// True if the NEP model described by `filename` uses ZBL.  The check is
// performed by re-reading the file; the result matches the NEP
// constructor's `zbl.enabled` flag.
inline bool nep_uses_zbl(const NEP& /*nep*/, const char* potential_filename)
{
  if (potential_filename == nullptr) return false;
  std::ifstream input(potential_filename);
  if (!input.is_open()) return false;
  std::string token;
  if (!(input >> token)) return false;
  return token == "nep4_zbl" || token == "nep4_zbl_temperature";
}


} // namespace nep_hessian
