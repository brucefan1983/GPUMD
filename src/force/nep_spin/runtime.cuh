/* Copyright 2017 Zheyong Fan and GPUMD development team
   This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "force/potential.cuh"
#include "mfield.cuh"
#include "model/atom.cuh"
#include <memory>
#include <string>

class Spin_Runtime
{
public:
  std::unique_ptr<Potential> create_potential(const std::vector<std::string>& tokens, int atoms);
  void initialize(Atom& atom) const;
  void check_command(const std::vector<std::string>& tokens);
  void validate(const Atom& atom) const;
  void parse_field(const std::vector<std::string>& tokens, const std::vector<Group>& group);
  void apply_field(
    Box& box,
    const GPU_Vector<double>& position,
    const std::vector<Group>& group,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& spin,
    GPU_Vector<double>& energy,
    GPU_Vector<double>& mforce) const;

private:
  void set_types(const std::vector<int>& types);
  std::string unsupported_;
  std::vector<int> types_;
  GPU_Vector<int> active_;
  Spin_Magnetic_Field field_;
};
