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

#pragma once
#include <jni.h>
#include "neighbor.cuh"
#include "potential.cuh"

class NNAP : public Potential
{
public:
  using Potential::compute;
  NNAP(const char* filename, int num_atoms);
  virtual ~NNAP(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);
    
protected:
  JNIEnv *mEnv = NULL;
  jobject mCore = NULL;
  Neighbor neighbor;
};
