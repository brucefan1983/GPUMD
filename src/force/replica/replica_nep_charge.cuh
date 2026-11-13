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
Charge NEP (qNEP) for replica dynamics.

One object per replica stream: its qNEP state (descriptors, charges,
Ewald or PPPM mesh and FFT plans) is only used by that stream, and all of its
device work runs on it. The candidate neighbor list of the large-box path is a
Replica_Neighbor on the same stream.
------------------------------------------------------------------------------*/

#pragma once
#include "replica_neighbor.cuh"
#include "replica_nep_charge_model.cuh"

class RunInput;

class NEP_Charge_Replica : public NEP_Charge_Replica_Model
{
public:
  NEP_Charge_Replica(
    const char* file_potential,
    const int num_atoms,
    const RunInput& run_input,
    gpuStream_t stream,
    const bool verbose);

protected:
  void update_candidate_neighbors(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position_per_atom,
    const int*& NN,
    const int*& NL) override;

private:
  Replica_Neighbor neighbor_;
};
