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
GPUMD Contributing author: Alexander Gabourie (Stanford University)
------------------------------------------------------------------------------*/

#pragma once
#include "property.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#ifdef USE_HIP
  #include <hipblas/hipblas.h>
#else
  #include <cublas_v2.h>
#endif
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define NO_METHOD -1
#define GKMA_METHOD 0
#define HNEMA_METHOD 1

class MODAL_ANALYSIS : public Property
{
public:
  // Bookkeeping variables
  int compute = 0;
  int method = NO_METHOD; // Method to compute
  int output_interval;    // number of times steps to output average heat current
  int sample_interval;    // steps per heat current computation
  int first_mode;         // first mode to consider
  int last_mode;          // last mode to consider
  int bin_size;           // number of modes per bin
  double f_bin_size;      // freq. range per bin (THz)
  int f_flag;             // 0 -> modes, 1 -> freq.
  int num_modes;          // total number of modes to consider
  int atom_begin;         // Beginning atom group/type
  int atom_end;           // End atom group/type

  // Data structures
  // eigenvectors x,y, and z
  GPU_Vector<float> eigx;
  GPU_Vector<float> eigy;
  GPU_Vector<float> eigz;

  // modal velocities
  GPU_Vector<float> xdotx;
  GPU_Vector<float> xdoty;
  GPU_Vector<float> xdotz;

  // precalculated mass values
  GPU_Vector<float> sqrtmass;
  GPU_Vector<float> rsqrtmass;

  // modal heat currents
  GPU_Vector<float> jmx;
  GPU_Vector<float> jmy;
  GPU_Vector<float> jmz;
  GPU_Vector<float> jm;

  GPU_Vector<float> bin_out; // modal binning structure
  GPU_Vector<int> bin_count; // Number of modes per bin when f_flag=1
  GPU_Vector<int> bin_sum;   // Running sum from bin_count

  char eig_file_position[200];
  char output_file_position[200];

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void process(
      const int number_of_steps,
      int step,
      const int fixed_group,
      const int move_group,
      const double global_time,
      const double temperature,
      Integrate& integrate,
      Box& box,
      std::vector<Group>& group,
      GPU_Vector<double>& thermo,
      Atom& atom,
      Force& force);

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

  MODAL_ANALYSIS(
    const char** param, 
    int num_param, 
    const int number_of_types, 
    int method_input,
    Force& force);
  void parse_compute_gkma(const char**, int, const int number_of_types);
  void parse_compute_hnema(const char**, int, const int number_of_types);

private:
  int samples_per_output; // samples to be averaged for output
  int num_bins;           // number of bins to output
  int N1;                 // Atom starting index
  int N2;                 // Atom ending index
  int num_participating;  // Number of particles participating
  int num_heat_stored;    // Number of stored heat current elements

  double fe_x = 0.0;
  double fe_y = 0.0;
  double fe_z = 0.0;
  double fe;

  gpublasHandle_t ma_handle;

  // stress by by square root mass (intermediate term)
  GPU_Vector<float> smx;
  GPU_Vector<float> smy;
  GPU_Vector<float> smz;

  // sqrt(mass)*velocity (intermediate term)
  GPU_Vector<float> mvx;
  GPU_Vector<float> mvy;
  GPU_Vector<float> mvz;

  void compute_heat(
    const GPU_Vector<double>& velocity_per_atom, const GPU_Vector<double>& virial_per_atom);

  void setN(const std::vector<int>& cpu_type_size);
  void set_eigmode(int, std::ifstream&, GPU_Vector<float>&);
};
