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

/*----------------------------------------------------------------------------80
Some functions for dealing with text files. Written by Mikko Ervasti.
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "read_file.cuh"
#include <ctype.h>
#include <errno.h>
#include <cstring>

int is_valid_int(const char* s, int* result)
{
  if (s == NULL) {
    return 0;
  } else if (*s == '\0') {
    return 0;
  }
  char* p;
  errno = 0;
  *result = (int)strtol(s, &p, 0);
  if (errno != 0 || s == p || *p != 0) {
    return 0;
  } else {
    return 1;
  }
}

int is_valid_real(const char* s, double* result)
{
  if (s == NULL) {
    return 0;
  } else if (*s == '\0') {
    return 0;
  }
  char* p;
  errno = 0;
  *result = strtod(s, &p);
  if (errno != 0 || s == p || *p != 0) {
    return 0;
  } else {
    return 1;
  }
}
