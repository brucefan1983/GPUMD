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

#pragma once
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    const cudaError_t error_code = call;                                                           \
    if (error_code != cudaSuccess) {                                                               \
      fprintf(stderr, "CUDA Error:\n");                                                            \
      fprintf(stderr, "    File:       %s\n", __FILE__);                                           \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                                           \
      fprintf(stderr, "    Error code: %d\n", error_code);                                         \
      fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code));                     \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define PRINT_SCANF_ERROR(count, n, text)                                                          \
  do {                                                                                             \
    if (count != n) {                                                                              \
      fprintf(stderr, "Input Error:\n");                                                           \
      fprintf(stderr, "    File:       %s\n", __FILE__);                                           \
      fprintf(stderr, "    Line:       %d\n", __LINE__);                                           \
      fprintf(stderr, "    Error text: %s\n", text);                                               \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define PRINT_INPUT_ERROR(text)                                                                    \
  do {                                                                                             \
    fprintf(stderr, "Input Error:\n");                                                             \
    fprintf(stderr, "    File:       %s\n", __FILE__);                                             \
    fprintf(stderr, "    Line:       %d\n", __LINE__);                                             \
    fprintf(stderr, "    Error text: %s\n", text);                                                 \
    exit(1);                                                                                       \
  } while (0)

#define PRINT_KEYWORD_ERROR(keyword)                                                               \
  do {                                                                                             \
    fprintf(stderr, "Input Error:\n");                                                             \
    fprintf(stderr, "    File:       %s\n", __FILE__);                                             \
    fprintf(stderr, "    Line:       %d\n", __LINE__);                                             \
    fprintf(stderr, "    Error text: '%s' is an invalid keyword.\n", keyword);                     \
    exit(1);                                                                                       \
  } while (0)

#ifdef STRONG_DEBUG
#define CUDA_CHECK_KERNEL                                                                          \
  {                                                                                                \
    CHECK(cudaGetLastError());                                                                     \
    CHECK(cudaDeviceSynchronize());                                                                \
  }
#else
#define CUDA_CHECK_KERNEL                                                                          \
  {                                                                                                \
    CHECK(cudaGetLastError());                                                                     \
  }
#endif

void print_line_1(void);
void print_line_2(void);
FILE* my_fopen(const char* filename, const char* mode);
std::vector<std::string> get_tokens(const std::string& line);
std::vector<std::string> get_tokens(std::ifstream& input);
int get_int_from_token(const std::string& token, const char* filename, const int line);
float get_float_from_token(const std::string& token, const char* filename, const int line);
