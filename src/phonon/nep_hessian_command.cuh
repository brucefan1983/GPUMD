/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include <string>
#include <vector>
#include <array>

class Atom;
class Box;
class Force;
class Group;

class NEP_Hessian_Command
{
public:
  void parse(const std::vector<std::string>& tokens);
  void compute(
    Force& force, Box& box, Atom& atom, std::vector<Group>& group,
    const std::string& potential_filename);

private:
  void compute_analytic(
    Force& force, Box& box, Atom& atom, std::vector<Group>& group,
    const std::string& potential_filename);

  std::string output_ = "hessian.out";
  std::string output_format_ = "dense";
  std::string raw_output_;
  std::string metadata_output_;
  std::string structure_output_;
  std::string phonon_mode_ = "none";
  std::array<int, 3> supercell_{{1, 1, 1}};
  std::string kpoints_file_ = "kpoints.in";
  int kpoint_intervals_ = 100;
};
