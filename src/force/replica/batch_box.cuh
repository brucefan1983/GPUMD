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

Replica r owns atoms [r * atoms_per_replica, (r + 1) * atoms_per_replica), and
all replicas share one box.
------------------------------------------------------------------------------*/

#pragma once
#include "model/box.cuh"

struct Batch_Box {
  Box box;
  int atoms_per_replica = 0;
};

// Kernels templated on the box view accept either a single Box or a Batch_Box.
__host__ __device__ inline const Box& box_of(const Box& box, const int) { return box; }
__host__ __device__ inline const Box& box_of(const Batch_Box& box, const int) { return box.box; }
