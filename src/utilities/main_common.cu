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

#include "error.cuh"
#include "main_common.cuh"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>

void print_compile_information(void)
{
  print_line_1();
  printf("Compiling options:\n");
  print_line_2();

#ifdef DEBUG
  printf("DEBUG is on: Use a fixed PRNG seed for different runs.\n");
#else
  srand(std::chrono::system_clock::now().time_since_epoch().count());
  printf("DEBUG is off: Use different PRNG seeds for different runs.\n");
#endif

#ifdef USE_FCP

#ifdef USE_NEP
  PRINT_INPUT_ERROR("Cannot add both -DUSE_FCP and -DUSE_NEP to the makefile.");
#else
  printf("This version can only be used with an FCP potential.\n");
#endif

#else

#ifdef USE_NEP
  printf("This version can only be used with an NEP potential.\n");
#else
  printf("This version can only be used with empirical potentials (not FCP or NEP).\n");
#endif

#endif
}

void print_gpu_information(void)
{
  print_line_1();
  printf("GPU information:\n");
  print_line_2();

  int device_id = 0;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, device_id));

  printf("Device id:               %d\n", device_id);
  printf("Device name:             %s\n", prop.name);
  printf("Compute capability:      %d.%d\n", prop.major, prop.minor);
  printf("Amount of global memory: %g GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
  printf("Number of SMs:           %d\n", prop.multiProcessorCount);
}

int get_number_of_input_directories(void)
{
  int number_of_inputs;
  int count = scanf("%d", &number_of_inputs);
  PRINT_SCANF_ERROR(count, 1, "Reading error for number of inputs.");
  return number_of_inputs;
}
