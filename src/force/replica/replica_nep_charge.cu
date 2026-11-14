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
Charge NEP for replica dynamics, see replica_nep_charge.cuh.
------------------------------------------------------------------------------*/

#include "replica_nep_charge.cuh"
#include "utilities/error.cuh"

NEP_Charge_Replica::NEP_Charge_Replica(
  const char* file_potential,
  const int num_atoms,
  const RunInput& run_input,
  const gpuStream_t stream,
  const bool verbose)
  : NEP_Charge_Replica_Model(file_potential, num_atoms, run_input, verbose)
{
  if (stream == nullptr)
    PRINT_INPUT_ERROR("Replica qNEP evaluation requires an explicit GPU stream.\n");
  stream_ = stream;
  ewald.set_stream(stream);
  pppm.set_stream(stream);
  neighbor_.initialize(rc, num_atoms, paramb.MN_radial);
  N1 = 0;
  N2 = num_atoms;
}

void NEP_Charge_Replica::update_candidate_neighbors(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  const int*& NN,
  const int*& NL)
{
  neighbor_.update(rc, box, type.size(), position_per_atom.data(), stream_);
  NN = neighbor_.NN.data();
  NL = neighbor_.NL.data();
}
