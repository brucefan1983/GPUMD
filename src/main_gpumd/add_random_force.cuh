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

#pragma once
#include "utilities/gpu_vector.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
  #include <hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif

class Atom;

class Add_Random_Force
{
public:
  void parse(const char** param, int num_param, int number_of_atoms);
  void compute(const int step, Atom& atom);
  void finalize();

private:
  GPU_Vector<gpurandState> curand_states_;
  int num_calls_ = 0;
  double force_variance_ = 0.0;
};
