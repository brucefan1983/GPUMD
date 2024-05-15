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

#include "fitness.cuh"
#include "parameters.cuh"
#include "snes.cuh"
#include "utilities/error.cuh"
#include "utilities/main_common.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_welcome_information(void);

int main(int argc, char* argv[])
{
  print_welcome_information();
  print_gpu_information();

  print_line_1();
  printf("Started running nep.\n");
  print_line_2();

  clock_t time_begin = clock();
  Parameters para;
  Fitness fitness(para);
  clock_t time_finish = clock();

  float time_used = (time_finish - time_begin) / float(CLOCKS_PER_SEC);
  print_line_1();
  printf("Time used for initialization = %f s.\n", time_used);
  print_line_2();

  time_begin = clock();
  SNES snes(para, &fitness);
  time_finish = clock();

  time_used = (time_finish - time_begin) / float(CLOCKS_PER_SEC);
  print_line_1();
  if (para.prediction == 0) {
    printf("Time used for training = %f s.\n", time_used);
  } else {
    printf("Time used for predicting = %f s.\n", time_used);
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
  printf("*                    Version 3.9.4                            *\n");
  printf("*              This is the nep executable                     *\n");
  printf("***************************************************************\n");
  printf("\n");
}
