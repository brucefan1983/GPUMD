/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include "gpu_macro.cuh"
#include "main_common.cuh"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

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
}

void print_gpu_information(void)
{
  print_line_1();
  printf("GPU information:\n");
  print_line_2();

  int num_gpus;
  CHECK(gpuGetDeviceCount(&num_gpus));
  printf("number of GPUs = %d\n", num_gpus);

  for (int device_id = 0; device_id < num_gpus; ++device_id) {
    gpuDeviceProp prop;
    CHECK(gpuGetDeviceProperties(&prop, device_id));

    printf("Device id:                   %d\n", device_id);
    printf("    Device name:             %s\n", prop.name);
    printf("    Compute capability:      %d.%d\n", prop.major, prop.minor);
    printf("    Amount of global memory: %g GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("    Number of SMs:           %d\n", prop.multiProcessorCount);
  }

  for (int i = 0; i < num_gpus; i++) {
    gpuSetDevice(i);
    for (int j = 0; j < num_gpus; j++) {
      int can_access;
      if (i != j) {
        CHECK(gpuDeviceCanAccessPeer(&can_access, i, j));
        if (can_access) {
          CHECK(gpuDeviceEnablePeerAccess(j, 0));
          printf("GPU-%d can access GPU-%d.\n", i, j);
        } else {
          printf("GPU-%d cannot access GPU-%d.\n", i, j);
        }
      }
    }
  }

  gpuSetDevice(0); // normally use GPU-0
}
