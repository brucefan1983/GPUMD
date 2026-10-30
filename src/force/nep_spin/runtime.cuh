/* Copyright 2017 Zheyong Fan and GPUMD development team
   This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "force/potential.cuh"
#include "model/atom.cuh"
#include <memory>
#include <string>

class RunInput;

class Spin_Runtime
{
public:
  explicit Spin_Runtime(const RunInput& input);
  std::unique_ptr<Potential> create_potential(const std::vector<std::string>& tokens, int atoms);
  void initialize(Atom& atom) const;

private:
  void check_command(const std::vector<std::string>& tokens) const;
  std::vector<int> types_;
};
