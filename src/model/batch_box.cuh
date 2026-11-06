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
Box view for a batch of replicas stored as one concatenated system.

Replica r owns atoms [r * atoms_per_replica, (r + 1) * atoms_per_replica).
All replicas share `box` unless `boxes` points to one device Box per replica
(reserved for variable-cell ensembles).
------------------------------------------------------------------------------*/

#pragma once
#include "model/box.cuh"

struct Batch_Box {
  Box box;
  const Box* boxes = nullptr;
  int atoms_per_replica = 0;

  __host__ __device__ const Box& box_of(const int atom) const
  {
    return boxes == nullptr ? box : boxes[atom / atoms_per_replica];
  }
  __host__ __device__ int replica_of(const int atom) const { return atom / atoms_per_replica; }
};

// Kernels templated on the box view accept either a single Box or a Batch_Box.
__host__ __device__ inline const Box& box_of(const Box& box, const int) { return box; }
__host__ __device__ inline const Box& box_of(const Batch_Box& box, const int atom)
{
  return box.box_of(atom);
}

// Range of candidate neighbor indices for atom n1: the whole [N1, N2) for a
// single system, and the atom's own replica (clipped to [N1, N2)) for a batch.
__host__ __device__ inline void neighbor_range(
  const Box&, const int, const int N1, const int N2, int& begin, int& end)
{
  begin = N1;
  end = N2;
}
__host__ __device__ inline void neighbor_range(
  const Batch_Box& box, const int n1, const int N1, const int N2, int& begin, int& end)
{
  const int first = box.replica_of(n1) * box.atoms_per_replica;
  begin = first > N1 ? first : N1;
  end = first + box.atoms_per_replica < N2 ? first + box.atoms_per_replica : N2;
}
