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

#include "potential.cuh"
#include "nep.cuh"
#include "nep_vdw.cuh"
#include "nep_charge.cuh"
#include "nep_charge_vdw.cuh"
#include "tnep.cuh"
#include "parameters.cuh"

std::unique_ptr<Potential> create_potential()
{
  std::unique_ptr<Parameters> model_parameters(new Parameters);
  Parameters& para = *model_parameters;
  std::unique_ptr<Potential> potential;
  if (para.model_type == 1 || para.model_type == 2) {
    potential.reset(new TNEP(std::move(model_parameters)));
  } else {
    if (para.charge_vdw) {
      potential.reset(new NEP_Charge_VDW(std::move(model_parameters)));
    } else if (para.charge_mode) {
      potential.reset(new NEP_Charge(std::move(model_parameters)));
    } else if (para.vdw) {
      potential.reset(new NEP_VDW(std::move(model_parameters)));
    } else {
      potential.reset(new NEP(std::move(model_parameters)));
    }
  }
  return potential;
}
