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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

void print_error(const char* str)
{
  printf("ERROR: %s", str);
  exit(EXIT_FAILURE);
}

void print_line_1(void)
{
  printf("\n");
  printf("---------------------------------------------------------------\n");
}

void print_line_2(void)
{
  printf("---------------------------------------------------------------\n");
  printf("\n");
}

FILE* my_fopen(const char* filename, const char* mode)
{
  FILE* fid = fopen(filename, mode);
  if (fid == NULL) {
    printf("Failed to open %s!\n", filename);
    printf("%s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  return fid;
}
