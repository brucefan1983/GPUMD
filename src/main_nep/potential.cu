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
#include "main_nep/nep/nep.cuh"
#include "main_nep/nep_vdw/nep_vdw.cuh"
#include "main_nep/nep_charge/nep_charge.cuh"
#include "main_nep/nep_charge_vdw/nep_charge_vdw.cuh"
#include "main_nep/tnep/tnep.cuh"
#include "parameter_input.cuh"

std::unique_ptr<Potential> create_potential()
{
  ParameterInput input;
  std::unique_ptr<Potential> potential;
  if (input.model_type == 1 || input.model_type == 2) {
    potential.reset(new TNEP(input));
  } else {
    if (input.charge_vdw) {
      potential.reset(new NEP_Charge_VDW(input));
    } else if (input.charge_mode) {
      potential.reset(new NEP_Charge(input));
    } else if (input.vdw) {
      potential.reset(new NEP_VDW(input));
    } else {
      potential.reset(new NEP(input));
    }
  }
  return potential;
}
