/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.

    Spin3 training shares its descriptor and derivative implementation with
    the production NEP_Spin runtime.
*/

#pragma once
#include "spin_parameters.cuh"
void read_spin_model(Spin_Parameters&, float*);
void write_spin_model(FILE*, Spin_Parameters&, const float*, int max_radial, int max_angular);
