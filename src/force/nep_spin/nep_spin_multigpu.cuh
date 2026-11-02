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
Multi-GPU NEP-spin potential interface and domain state.

Contributors:
Chengbing Chen, 1747193328@qq.com
------------------------------------------------------------------------------*/

#pragma once
#include "nep_spin.cuh"
#include <memory>

class NEP_Spin_MultiGPU : public Potential
{
public:
  NEP_Spin_MultiGPU(int devices, const char* model, int atoms, int direction);
  ~NEP_Spin_MultiGPU() override;
  using Potential::compute;
  void compute(
    Box&,
    const GPU_Vector<int>&,
    const GPU_Vector<double>&,
    GPU_Vector<double>&,
    GPU_Vector<double>&,
    GPU_Vector<double>&) override
  {
    PRINT_INPUT_ERROR("NEP_Spin requires spin inputs.");
  }
  void compute(
    Box&,
    const GPU_Vector<int>&,
    const GPU_Vector<double>&,
    const GPU_Vector<double>&,
    GPU_Vector<double>&,
    GPU_Vector<double>&,
    GPU_Vector<double>&,
    GPU_Vector<double>&);
  const std::vector<int>& spin_types() const { return domains_[0]->potential->spin_types(); }

private:
  struct Domain {
    std::unique_ptr<NEP_Spin> potential;
    GPU_Vector<int> type;
    GPU_Vector<double> position, spin, energy, force, virial, mforce;
    int start = 0, size = 0, first = 0, owned = 0, global_first = 0;
    void resize(int count);
  };
  int devices_, atoms_, direction_;
  std::vector<std::unique_ptr<Domain>> domains_;
  GPU_Vector<int> cell_count_, cell_sum_, order_, packed_type_;
  GPU_Vector<double> packed_position_, packed_spin_, collected_;
  std::vector<int> prefix_;
  GPU_Vector<double> partition_position_;
  GPU_Vector<int> partition_changed_;
  double partition_box_[9] = {};
  bool partition_valid_ = false;
  gpuStream_t partition_stream_ = nullptr;
};
