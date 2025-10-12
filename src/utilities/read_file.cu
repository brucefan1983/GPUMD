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

static std::string get_potential_file_name()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string potential_file_name;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "potential") {
        potential_file_name = tokens[1];
        break;
      }
    }
  }

  input_run.close();
  return potential_file_name;
}

bool check_is_nep_charge()
{
  bool is_nep_charge = false;
  std::string potential_file_name = get_potential_file_name();

  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens[0].size() >= 12) {
    if (tokens[0].substr(0, 11) == "nep4_charge") {
      is_nep_charge = true;
    }
  } 
  if (tokens[0].size() >= 16) {
    if (tokens[0].substr(0, 15) == "nep4_zbl_charge") {
      is_nep_charge = true;
    }
  }
  input_potential.close();

  return is_nep_charge;
}

bool check_need_peratom_virial()
{
  bool need_peratom_virial = false;
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "compute_hac" || 
        tokens[0] == "compute_hnemd" || 
        tokens[0] == "compute_hnemdec" || 
        tokens[0] == "compute_shc" ||
        tokens[0] == "compute_gkma" ||
        tokens[0] == "compute_hnema") {
        need_peratom_virial = true;
        break;
      }
    }
  }
  input_run.close();
  return need_peratom_virial;
}
