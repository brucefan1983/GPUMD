/* Copyright 2017 Zheyong Fan and GPUMD development team
   This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#include "nep_spin.cuh"
#include "nep_spin_multigpu.cuh"
#include "runtime.cuh"
#include "utilities/error.cuh"
#include "utilities/run_input.cuh"
#include <cmath>

Spin_Runtime::Spin_Runtime(const RunInput& input)
{
  for (const auto& line : input.lines()) {
    if (!line.tokens.empty()) {
      check_command(line.tokens);
    }
  }
}

void Spin_Runtime::check_command(const std::vector<std::string>& tokens) const
{
  bool unsupported = false;
  for (const char* name :
       {"dump_observer",
        "compute_hac",
        "compute_hnemd",
        "compute_hnemdec",
        "compute_shc",
        "compute_gkma",
        "compute_hnema",
        "deform",
        "compute_lsqt",
        "minimize",
        "compute_phonon",
        "compute_cohesive",
        "compute_elastic",
        "kspace",
        "dftd3"}) {
    unsupported = unsupported || tokens[0] == name;
  }
  if (tokens[0] == "ensemble") {
    unsupported = tokens.size() < 2 || (tokens[1] != "nve" && tokens[1] != "nvt_nhc");
  }
  if (unsupported) {
    PRINT_INPUT_ERROR(("NEP-spin does not support command " + tokens[0] + ".").c_str());
  }
}

void Spin_Runtime::initialize(Atom& atom) const
{
  if (types_.empty())
    return;
  if (!atom.has_spin)
    PRINT_INPUT_ERROR("NEP-spin requires spin:R:3 in model.xyz.");
  const int n = atom.number_of_atoms;
  for (int i = 0; i < n; ++i) {
    const int type = atom.cpu_type[i];
    for (int d = 0; d < 3; ++d) {
      double& spin = atom.cpu_spin_per_atom[i + n * d];
      if (!types_[type])
        spin = 0.0;
      else if (!std::isfinite(spin))
        PRINT_INPUT_ERROR("Active spin components must be finite.");
    }
  }
  atom.spin_per_atom.copy_from_host(atom.cpu_spin_per_atom.data());
}

std::unique_ptr<Potential>
Spin_Runtime::create_potential(const std::vector<std::string>& tokens, int atoms)
{
  int direction = -1;
  if (tokens.size() == 3) {
    if (tokens[2] == "x")
      direction = 0;
    else if (tokens[2] == "y")
      direction = 1;
    else if (tokens[2] == "z")
      direction = 2;
    else
      PRINT_INPUT_ERROR("partition direction for multi-GPU NEP can only be x or y or z.");
  }
  int devices;
  CHECK(gpuGetDeviceCount(&devices));
  if (devices == 1) {
    std::unique_ptr<NEP_Spin> potential(new NEP_Spin(tokens[1].c_str(), atoms));
    types_ = potential->spin_types();
    return std::unique_ptr<Potential>(std::move(potential));
  }
  std::unique_ptr<NEP_Spin_MultiGPU> potential(
    new NEP_Spin_MultiGPU(devices, tokens[1].c_str(), atoms, direction));
  types_ = potential->spin_types();
  return std::unique_ptr<Potential>(std::move(potential));
}
