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

/*----------------------------------------------------------------------------80
Runtime interface for initializing and evaluating NEP-spin potentials.

Contributors:
Chengbing Chen, 1747193328@qq.com
------------------------------------------------------------------------------*/

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
  void compute(Potential& potential, Box& box, Atom& atom) const;

private:
  void check_command(const std::vector<std::string>& tokens) const;
  std::vector<int> types_;
};
