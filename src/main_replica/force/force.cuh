/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <map>
#include <memory>
#include <string>
#include <vector>

class NEP;
class NEP_Charge;

class Force
{
public:
  Force();
  ~Force();

  void
  parse_potential(const char** param, int num_param, const Box& box, const int number_of_atoms);

  void clear();
  bool has_potential() const;

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void prepare_stream(const gpuStream_t stream);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    const gpuStream_t stream);

private:
  enum class Model_Type { none, energy_nep, charge_nep };

  Model_Type model_type_ = Model_Type::none;
  std::unique_ptr<NEP> potential_;
  std::map<gpuStream_t, std::unique_ptr<NEP_Charge>> charge_potentials_;
  std::string potential_file_;
  int number_of_atoms_ = 0;
};
