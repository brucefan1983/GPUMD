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

#include "fitness.cuh"
#include "parameters.cuh"
#include "snes.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <cstring>

void print_welcome_information(void);

int main(int argc, char* argv[])
{
  print_welcome_information();
  print_gpu_information();

  print_line_1();
  printf("Started running nep.\n");
  print_line_2();

  const auto time_begin1 = std::chrono::high_resolution_clock::now();
  Parameters para;
  Fitness fitness(para);
  const auto time_finish1 = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double> time_used1 = time_finish1 - time_begin1;
  print_line_1();
  printf("Time used for initialization = %f s.\n", time_used1.count());
  print_line_2();

  const auto time_begin2 = std::chrono::high_resolution_clock::now();
  SNES snes(para, &fitness);
  const auto time_finish2 = std::chrono::high_resolution_clock::now();

  const std::chrono::duration<double> time_used2 = time_finish2 - time_begin2;
  print_line_1();
  if (para.prediction == 0) {
    printf("Time used for training = %f s.\n", time_used2.count());
  } else {
    printf("Time used for predicting = %f s.\n", time_used2.count());
  }

  print_line_2();

  print_line_1();
  printf("Finished running nep.\n");
  print_line_2();

  return EXIT_SUCCESS;
}

void print_welcome_information(void)
{
  printf("\n");
  printf("***************************************************************\n");
  printf("*                 Welcome to use GPUMD                        *\n");
  printf("*    (Graphics Processing Units Molecular Dynamics)           *\n");
  printf("*                     version 4.3                             *\n");
  printf("*              This is the nep executable                     *\n");
  printf("***************************************************************\n");
  printf("\n");
}
