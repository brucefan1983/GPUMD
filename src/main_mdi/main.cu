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

#include "run.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include <chrono>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef USE_MDI
extern "C" int mdi_engine_main(struct Run* run, int argc, char* argv[]);
#endif

void print_welcome_information();

int main(int argc, char* argv[])
{
  const char* run_input = "run.in";
  for (int i = 1; i < argc - 1; ++i) {
    if (std::strcmp(argv[i], "-in") == 0) {
      run_input = argv[i + 1];
      break;
    }
  }

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mdi") == 0) {
#ifdef USE_MDI
      Run run_for_mdi(true, run_input); // skip run commands - MDI will control stepping
      return mdi_engine_main(&run_for_mdi, argc, argv);
#else
      printf("MDI support not enabled at build time. Rebuild with USE_MDI=1.\n");
      return EXIT_FAILURE;
#endif
    }
  }
  print_welcome_information();
  print_compile_information();
  print_gpu_information();

  print_line_1();
  printf("Started running GPUMD.\n");
  print_line_2();

  CHECK(gpuDeviceSynchronize());
  const auto time_begin = std::chrono::high_resolution_clock::now();

  Run run(false, run_input);

  CHECK(gpuDeviceSynchronize());
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  print_line_1();
  printf("Time used = %f s.\n", time_used.count());
  print_line_2();

  print_line_1();
  printf("Finished running GPUMD.\n");
  print_line_2();

  return EXIT_SUCCESS;
}

void print_welcome_information(void)
{
  printf("\n");
  printf("***************************************************************\n");
  printf("*                 Welcome to use GPUMD                        *\n");
  printf("*     (Graphics Processing Units Molecular Dynamics)          *\n");
  printf("*                    version 4.9.1                            *\n");
  printf("*              This is the gpumd executable                   *\n");
  printf("***************************************************************\n");
  printf("\n");
}
