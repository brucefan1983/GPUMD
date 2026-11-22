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
Some functions for dealing with text files.

Major contributors:
  Mikko Ervasti: Original text-file reader and line tokenizer.
  Zheyong Fan: Integrated numeric input validation and added charge-model detection.
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "read_file.cuh"
#include "utilities/nep_io/nep_file.cuh"
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <errno.h>
#include <fstream>

int is_valid_int(const char* s, int* result)
{
  if (s == NULL) {
    return 0;
  } else if (*s == '\0') {
    return 0;
  }
  char* p;
  errno = 0;
  const long value = strtol(s, &p, 0);
  if (errno != 0 || s == p || *p != 0 || value < INT_MIN || value > INT_MAX) {
    return 0;
  }
  *result = static_cast<int>(value);
  return 1;
}

int is_valid_int(const std::string& s, int* result) { return is_valid_int(s.c_str(), result); }

int is_valid_real(const char* s, double* result)
{
  if (s == NULL) {
    return 0;
  } else if (*s == '\0') {
    return 0;
  }
  char* p;
  errno = 0;
  const double value = strtod(s, &p);
  if (errno != 0 || s == p || *p != 0 || !std::isfinite(value)) {
    return 0;
  }
  *result = value;
  return 1;
}

int is_valid_real(const std::string& s, double* result) { return is_valid_real(s.c_str(), result); }

bool is_nep_charge_potential(const std::string& potential_file_name)
{
  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens.empty()) {
    PRINT_INPUT_ERROR("Empty potential file.");
  }
  if (NEP_File::is_nep_header(tokens[0])) {
    input_potential.clear();
    input_potential.seekg(0);
    std::string model;
    std::string error;
    if (!NEP_File::read_model_name(input_potential, model, error)) {
      PRINT_INPUT_ERROR(error.c_str());
    }
    return model == "qnep1" || model == "qnep2" || model == "qnep3" || model == "qnep3_vdw";
  }
  return false;
}
