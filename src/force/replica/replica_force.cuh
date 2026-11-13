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
Force evaluation of the replicas on one GPU.

Replica dynamics supports one energy NEP or qNEP model. Each replica is
evaluated on its own stream: an energy NEP keeps one model per GPU with one
workspace per stream, and a qNEP keeps one model per stream.
------------------------------------------------------------------------------*/

#pragma once
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <map>
#include <memory>
#include <string>

class NEP_Replica;
class NEP_Charge_Replica;
class RunInput;

class Replica_Force
{
public:
  Replica_Force();
  ~Replica_Force();

  // potential_file is the file of the potential command of run.in; verbose prints the
  // model information (once for all GPUs).
  void initialize(
    const std::string& potential_file,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input,
    const bool verbose);
  void clear();
  bool is_charge_nep() const { return model_type_ == Model_Type::charge_nep; }

  // Allocates the per-stream state; compute() requires it.
  void prepare_stream(const gpuStream_t stream);

  // Wraps positions into the box and evaluates potential, force and virial on a stream.
  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    const gpuStream_t stream);

private:
  enum class Model_Type { none, energy_nep, charge_nep };
  Model_Type model_type_ = Model_Type::none;
  std::unique_ptr<NEP_Replica> energy_nep_;
  std::map<gpuStream_t, std::unique_ptr<NEP_Charge_Replica>> charge_neps_;
  std::string potential_file_;
  int number_of_atoms_ = 0;
  const RunInput* run_input_ = nullptr;
  bool verbose_ = false;
};
