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

#pragma once
#include "dataset.cuh"
#include "potential.cuh"
#include "nep_compile.cuh"
#include "nep_evaluate.cuh"
#include <memory>
class Parameters;

class NEP : public Potential
{
public:
  struct ZBL {
    bool enabled = false;
    bool flexible = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  NEP() = default;
  void initialize(Parameters& para, int N, int Nc, int deviceCount) override;
  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount) override;

private:
  NEP_Neighbor neighbor[16];
  NEP_Descriptor_Parameters paramb;
  NEP_ANN annmb[16];
  NEP_Data nep_data[16];
  ZBL zbl;
  std::unique_ptr<NEP_Compile> compiled_kernel_;
  void initialize_model(const Parameters& para);
  void initialize_device(const Parameters& para, int N, int device_id);
  void initialize_compiled_kernel(const Parameters& para);
};
