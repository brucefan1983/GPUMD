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
Bound check of the neighbor lists.

A kernel that builds a neighbor list writes an entry only while the count of
the atom is below the capacity of the list, keeps counting beyond it, and
records the atom with record_neighbor_overflow. The host calls
check_neighbor_overflow after the kernel and before any kernel reads the list.

Every translation unit that includes this header holds its own record.
------------------------------------------------------------------------------*/

#pragma once
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <climits>
#include <stdio.h>
#include <stdlib.h>

enum NeighborListKind {
  NEIGHBOR_LIST_CANDIDATE,
  NEIGHBOR_LIST_LOCAL,
  NEIGHBOR_LIST_RADIAL,
  NEIGHBOR_LIST_ANGULAR,
  NUMBER_OF_NEIGHBOR_LIST_KINDS
};

// Per kind: the lowest index of an atom beyond the capacity, the largest count, the capacity.
// The run stops at the first overflow, so the record is never reset.
static __device__ int neighbor_overflow_record[NUMBER_OF_NEIGHBOR_LIST_KINDS][3] = {
  {INT_MAX, 0, 0}, {INT_MAX, 0, 0}, {INT_MAX, 0, 0}, {INT_MAX, 0, 0}};

static __device__ __forceinline__ void
record_neighbor_overflow(const int kind, const int atom, const int count, const int capacity)
{
  atomicMin(&neighbor_overflow_record[kind][0], atom);
  atomicMax(&neighbor_overflow_record[kind][1], count);
  neighbor_overflow_record[kind][2] = capacity;
}

static inline void check_neighbor_overflow()
{
  const char* names[NUMBER_OF_NEIGHBOR_LIST_KINDS] = {"candidate", "local", "radial", "angular"};
  int record[NUMBER_OF_NEIGHBOR_LIST_KINDS][3];
  CHECK(gpuMemcpyFromSymbol(record, neighbor_overflow_record, sizeof(record)));
  for (int kind = 0; kind < NUMBER_OF_NEIGHBOR_LIST_KINDS; ++kind) {
    if (record[kind][1] > 0) {
      fprintf(stderr, "Error:\n");
      fprintf(
        stderr,
        "    The %s neighbor list holds %d neighbors per atom, but atom %d has more.\n",
        names[kind],
        record[kind][2],
        record[kind][0]);
      fprintf(stderr, "    The largest number of neighbors of an atom is %d.\n", record[kind][1]);
      exit(1);
    }
  }
}
