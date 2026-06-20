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

/**
 * NNAP interface for GPUMD
 * @author liqa
 */

#ifdef USE_NNAP
#pragma once
#include <jni.h>
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"

class NNAP : public Potential
{
public:
  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types = 0;

    bool use_typewise_cutoff = false;
    float typewise_cutoff_factor = 0.0f;
  };

  using Potential::compute;
  NNAP(const char* setting_file, const char* nnap_file, int num_atoms);
  virtual ~NNAP(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
    
private:
  ZBL zbl;

  JNIEnv *mEnv = NULL;
  jobject mCore = NULL;
  Neighbor neighbor;
  
  GPU_Vector<float> nl_dx;
  GPU_Vector<float> nl_dy;
  GPU_Vector<float> nl_dz;
};
#endif
