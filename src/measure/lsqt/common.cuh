/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef USE_SP
typedef float real; // single precision
#else
typedef double real; // double precision
#endif

#ifndef CPU_ONLY
#include <stdio.h>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    const cudaError_t error_code = call;                                                           \
    if (error_code != cudaSuccess) {                                                               \
      printf("CUDA Error:\n");                                                                     \
      printf("    File:       %s\n", __FILE__);                                                    \
      printf("    Line:       %d\n", __LINE__);                                                    \
      printf("    Error code: %d\n", error_code);                                                  \
      printf("    Error text: %s\n", cudaGetErrorString(error_code));                              \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#endif // #ifndef CPU_ONLY
