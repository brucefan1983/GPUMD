/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details. You should have received a copy of the GNU General
    Public License along with GPUMD. If not, see <http://www.gnu.org/licenses/>.
*/

#include "read_file.cuh"
#include <cerrno>
#include <cstdlib>

int is_valid_int(const char* value, int* result)
{
  if (value == nullptr || *value == '\0')
    return 0;

  char* end;
  errno = 0;
  *result = static_cast<int>(strtol(value, &end, 0));
  return errno == 0 && value != end && *end == '\0';
}

int is_valid_real(const char* value, double* result)
{
  if (value == nullptr || *value == '\0')
    return 0;

  char* end;
  errno = 0;
  *result = strtod(value, &end);
  return errno == 0 && value != end && *end == '\0';
}
