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
#include <cufft.h>

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
    int K0K1K2 = 4096;      // total number of mesh points
    int K0K1;               // K[0] * K[1]
    int K[3];               // number of mesh points in the box vector directions
    int K_half[3];          // K/2
    float alpha_factor;     // 1 / (4 * alpha * alpha)
    float two_pi_over_V;    // 2 * pi / volume
    float volume_per_cell;  // volume / K1K2K3
    float b[3][3];          // b-vectors in reciprocal space
    float two_pi_over_K[3]; // 2 * pi ./ K
  };
private:
  Para para;
  int num_kpoints_max = 1;
  float alpha = 0.5f; // 1 / (2 Angstrom)
  float alpha_factor = 1.0f; // 1 / (4 * alpha * alpha)
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<cufftComplex> mesh;
  GPU_Vector<cufftComplex> mesh_fft;
  GPU_Vector<cufftComplex> mesh_fft_x;
  GPU_Vector<cufftComplex> mesh_fft_y;
  GPU_Vector<cufftComplex> mesh_fff_z;
  GPU_Vector<cufftComplex> mesh_fft_x_ifft;
  GPU_Vector<cufftComplex> mesh_fft_y_ifft;
  GPU_Vector<cufftComplex> mesh_fft_z_ifft;
  void allocate_memory();
  void find_para(const Box& box);
  void find_k_and_G(const double* box);
};
