/* Copyright 2017 Zheyong Fan and GPUMD development team
   This file is part of GPUMD. SPDX-License-Identifier: LGPL-3.0-or-later */
#pragma once
#include "nep.cuh"

// Structural force evaluation shared by NEP and NEP-spin. CUDA kernels remain internal.
void launch_find_force_radial(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  int N,
  int N1,
  int N2,
  Box box,
  const int* NN,
  const int* NL,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  const float* Fp,
  double* fx,
  double* fy,
  double* fz,
  double* virial);
void launch_find_partial_force_angular(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  int N,
  int N1,
  int N2,
  Box box,
  const int* NN_angular,
  const int* NL_angular,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  const float* Fp,
  const float* sum_fxyz,
  float* f12x,
  float* f12y,
  float* f12z);
