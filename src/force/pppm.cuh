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
#include "model/box.cuh"
#ifdef USE_HIP
  #include <hipfft/hipfft.h>
#else
  #include <cufft.h>
#endif

class PPPM
{
public:
  PPPM();
  ~PPPM();
  void initialize(const float alpha_input);
  void find_force(
    const int N,
    const int N1,
    const int N2,
    const Box& box,
    const GPU_Vector<float>& charge,
    const GPU_Vector<double>& position_per_atom,
    GPU_Vector<float>& D_real,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& potential_per_atom);
  struct Para {
    int K0K1K2;             // total number of mesh points
    int K0K1;               // K[0] * K[1]
    int K[3];               // number of mesh points in the box vector directions
    int K_half[3];          // K/2
    float alpha;            // The Ewald parameter
    float alpha_factor;     // 1 / (4 * alpha * alpha)
    float two_pi_over_V;    // 4pi/(2V)
    float potential_factor; // K_C_SP / N
    float b[3][3];          // b-vectors in reciprocal space
    float two_pi_over_K[3]; // 2 * pi ./ K
  };
private:
  Para para;
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<gpufftComplex> mesh;
  GPU_Vector<gpufftComplex> mesh_G;
  GPU_Vector<gpufftComplex> mesh_x;
  GPU_Vector<gpufftComplex> mesh_y;
  GPU_Vector<gpufftComplex> mesh_z;
  gpufftHandle plan;
  void allocate_memory();
  void find_para(const int N, const Box& box);
  void find_k_and_G(const double* box);

  bool need_peratom_virial = false;
  GPU_Vector<gpufftComplex> mesh_virial;
  gpufftHandle plan_virial;
};
